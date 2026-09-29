"""Aggregate frozen development buildings without conflating visibility and reconstruction."""

# Long HTML template lines preserve readable markup.
# ruff: noqa: E501

import argparse
import hashlib
import html
import json
import os
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageOps

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from make_active_head_report import AMBER, BG, MINT, MUTED, PANEL, TEXT, font
from make_multiroom_report import top_down_image

METRICS = (
    "visible_edge_coverage",
    "edge_precision_10cm",
    "edge_completeness_10cm",
    "typed_edge_precision_10cm",
    "typed_edge_completeness_10cm",
)
COLORS = [MINT, AMBER, "#76baff", "#ec8db7"]
ROOT = Path(__file__).resolve().parents[1]
VISUAL_AUDIT = {
    "compact_apartment": {
        "status": "inspected_with_scene_defect",
        "observed_frames": ["F0004", "F0036"],
        "findings": [
            "Imported living-room plant foliage intersects the sofa arm/cushion. "
            "The frozen scene and acquired images are retained; furnishing placement needs "
            "mesh collision validation."
        ],
    }
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def frozen_suite(suite_path, output_root):
    """The display declaration must be the exact input frozen by the runner."""
    suite = read_json(suite_path)
    provenance_path = output_root / "suite.json"
    provenance = read_json(provenance_path)
    if provenance is None or provenance.get("suite") != suite:
        raise ValueError("Suite declaration differs from frozen run provenance")
    relative_path = suite_path.resolve().relative_to(ROOT).as_posix()
    if provenance["input_sha256"].get(relative_path) != sha256(suite_path):
        raise ValueError("Suite file bytes differ from the frozen input")
    for case in suite["cases"]:
        experiment = read_json(output_root / case["id"] / "run/experiment.json")
        if experiment is None:
            continue
        if experiment["config"] != provenance["case_configs"][case["id"]]:
            raise ValueError(f"Case config differs from frozen suite: {case['id']}")
        if experiment["checkpoint_sha256"] != provenance["input_sha256"][suite["checkpoint"]]:
            raise ValueError(f"Case checkpoint differs from frozen suite: {case['id']}")
    return suite, sha256(provenance_path)


def load_case(spec, root):
    """Retain every declared case; measured rows require matching frozen provenance."""
    directory = root / spec["id"]
    status = read_json(directory / "status.json", {})
    receipt = read_json(directory / "receipt.json")
    if status.get("status") == "complete" and not receipt:
        raise ValueError(f"Completed case is missing its frozen artifact receipt: {spec['id']}")
    for name, digest in (receipt or {}).get("output_sha256", {}).items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or sha256(path) != digest:
            raise ValueError(f"Case artifact differs from its frozen receipt: {spec['id']}/{name}")
    artifacts = [
        name
        for name in ("report.html", "replay.mp4", "poster.jpg", "overview.jpg", "cloud.json")
        if (directory / "report" / name).is_file()
    ]
    result = {
        "id": spec["id"],
        "title": spec["title"],
        "description": spec.get("description", ""),
        "split": spec.get("split", "development"),
        "status": status.get("status", "not_run"),
        "stage": status.get("stage"),
        "full_pipeline_wall_seconds": status.get("wall_seconds"),
        "evaluated": False,
        "available_artifacts": artifacts,
        "media_sha256": {name: sha256(directory / "report" / name) for name in artifacts},
        "receipt_sha256": sha256(directory / "receipt.json") if receipt else None,
        "visual_audit": VISUAL_AUDIT.get(spec["id"]),
    }
    evaluation_path = directory / "run/evaluation.json"
    experiment_path = directory / "run/experiment.json"
    evaluation = read_json(evaluation_path)
    if evaluation is None:
        return result
    if evaluation.get("status") != "complete":
        raise ValueError(f"Incomplete evaluation: {spec['id']}")
    if evaluation["provenance"]["experiment_sha256"] != sha256(experiment_path):
        raise ValueError(f"Frozen experiment changed: {spec['id']}")
    reference_path = directory / "capture/reference.json"
    if evaluation["provenance"]["reference_sha256"] != sha256(reference_path):
        raise ValueError(f"Frozen reference changed: {spec['id']}")
    experiment = read_json(experiment_path)
    if experiment.get("status") != "complete":
        raise ValueError(f"Evaluation references an incomplete experiment: {spec['id']}")
    final, visits, audit = evaluation["final"], evaluation["visits"], evaluation["path_audit"]
    length = sum(item["reference_edge_length_m"] for item in final["per_region"].values())
    if not np.isfinite(length) or length <= 0:
        raise ValueError(f"Invalid reference edge denominator: {spec['id']}")
    result.update(
        {
            "evaluated": True,
            "experiment_status": experiment["status"],
            "reference_edge_length_m": length,
            "reference_portal_count": len(read_json(reference_path)["portals"]),
            "unentered_reference_rooms": [
                room["id"]
                for room in read_json(reference_path)["rooms"]
                if room["kind"] == "room" and room["id"] not in visits["region_entry_order"]
            ],
            "predicted_edge_voxels": final["predicted_edge_voxels"],
            "metrics": {key: final[key] for key in METRICS},
            "frames": evaluation["frames"],
            "stations": evaluation["stations"],
            "path_length_m": evaluation["path_length_m"],
            "action_seconds": evaluation["action_seconds"],
            "wall_seconds": evaluation["wall_seconds"],
            "stop_reason": evaluation["stop_reason"],
            "visits": {
                key: visits[key]
                for key in (
                    "rooms_entered",
                    "reference_room_count",
                    "region_entry_order",
                    "region_transition_sequence",
                    "unique_portals_crossed",
                )
            },
            "all_reference_rooms_entered": visits["rooms_entered"]
            == visits["reference_room_count"],
            "path_audit": {
                key: audit[key]
                for key in (
                    "footprint_radius_m",
                    "architecture",
                    "furniture_recipe_aabb_proxy",
                    "all_motions_in_observed_safe_map",
                    "limitations",
                )
            },
            "timings": evaluation["timings"],
            "per_region": final["per_region"],
            "visibility_curve": evaluation["visibility_curve"],
            "provenance": {
                **evaluation["provenance"],
                "evaluation_sha256": sha256(evaluation_path),
                "checkpoint_sha256": experiment["checkpoint_sha256"],
            },
            "replay_check": read_json(directory / "run/replay_check.json"),
        }
    )
    return result


def aggregate(cases):
    """Use explicit denominators; unavailable cases are never invented as measured zeros."""
    measured = [case for case in cases if case["evaluated"]]
    edge_length = sum(case["reference_edge_length_m"] for case in measured)
    predicted_voxels = sum(case["predicted_edge_voxels"] for case in measured)
    macro, weighted = {}, {}
    for metric in METRICS:
        valid = [case for case in measured if case["metrics"][metric] is not None]
        macro[metric] = {
            "value": float(np.mean([case["metrics"][metric] for case in valid])) if valid else None,
            "denominator_buildings": len(valid),
        }
        denominator = (
            "predicted_edge_voxels" if "precision" in metric else "reference_edge_length_m"
        )
        weight = sum(case[denominator] for case in valid)
        weighted[metric] = {
            "value": sum(case["metrics"][metric] * case[denominator] for case in valid) / weight
            if weight
            else None,
            "denominator": denominator,
            "denominator_value": weight,
            "evaluated_buildings": len(valid),
        }
    return {
        "declared_buildings": len(cases),
        "evaluated_buildings": len(measured),
        "unevaluated_case_ids": [case["id"] for case in cases if not case["evaluated"]],
        "all_reference_rooms_entered_buildings": sum(
            case["all_reference_rooms_entered"] for case in measured
        ),
        "total_reference_rooms_entered": sum(case["visits"]["rooms_entered"] for case in measured),
        "total_reference_rooms": sum(case["visits"]["reference_room_count"] for case in measured),
        "reference_edge_length_m": edge_length,
        "predicted_edge_voxels": predicted_voxels,
        "total_frames": sum(case["frames"] for case in measured),
        "total_path_length_m": sum(case["path_length_m"] for case in measured),
        "total_wall_seconds": sum(case["wall_seconds"] for case in measured),
        "total_full_pipeline_wall_seconds": sum(
            case.get("full_pipeline_wall_seconds") or 0 for case in cases
        ),
        "full_pipeline_timing_buildings": sum(
            case.get("full_pipeline_wall_seconds") is not None for case in cases
        ),
        "macro": macro,
        "weighted": weighted,
        "missing_case_policy": "Unavailable measurements remain null and visible; aggregate denominators include evaluated buildings only. Run completion does not mean whole-building reconstruction.",
    }


def make_plots(cases, output):
    plt.rcParams.update(
        {
            "text.color": TEXT,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "font.size": 11,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.5), facecolor=BG, constrained_layout=True)
    for ax, field, label in zip(
        axes,
        ("path_length_m", "action_seconds"),
        ("Body travel (m)", "Simulated action time (s)"),
        strict=True,
    ):
        ax.set_facecolor(PANEL)
        for case, color in zip(cases, COLORS, strict=False):
            if not case["evaluated"]:
                continue
            curve = case["visibility_curve"]
            ax.step(
                [point[field] for point in curve],
                [100 * point["visible_edge_coverage"] for point in curve],
                where="post",
                color=color,
                label=case["id"].replace("_", " ").title(),
                linewidth=2.2,
            )
        ax.set(xlabel=label, ylabel="Reference edge visibility (%)", ylim=(0, 100))
        ax.grid(alpha=0.13)
        for spine in ax.spines.values():
            spine.set_color("#33465a")
    axes[0].legend(
        facecolor=BG, edgecolor="#33465a", labelcolor=TEXT, fontsize=9, loc="lower right"
    )
    fig.savefig(output / "coverage.png", dpi=150, facecolor=BG)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(13.5, 4.6), facecolor=BG, constrained_layout=True)
    ax.set_facecolor(PANEL)
    labels = (
        "Visibility",
        "3D precision",
        "3D completeness",
        "Typed precision",
        "Typed completeness",
    )
    x = np.arange(len(cases))
    width = 0.15
    for index, (metric, label, color) in enumerate(
        zip(METRICS, labels, [MINT, AMBER, "#76baff", "#ec8db7", "#bca2ff"], strict=True)
    ):
        values = [100 * case["metrics"][metric] if case["evaluated"] else np.nan for case in cases]
        ax.bar(x + (index - 2) * width, values, width, label=label, color=color)
    ax.set_xticks(x, [case["id"].replace("_", " ").title() for case in cases], fontsize=10)
    ax.set(
        ylabel="Percent",
        ylim=(0, 106),
        xlim=(-0.55, len(cases) - 0.45),
        title="Seeing structure and reconstructing it are different measurements",
    )
    ax.legend(loc="upper center", ncols=3, facecolor=BG, labelcolor=TEXT, fontsize=9)
    ax.grid(axis="y", alpha=0.13)
    fig.savefig(output / "comparison.png", dpi=150, facecolor=BG)
    plt.close(fig)


def make_contact_sheet(cases, root, output):
    width, row_height = 1800, 360
    canvas = Image.new("RGB", (width, 100 + len(cases) * row_height), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((25, 20), "ROOMGRAPH / FOUR DEVELOPMENT SPACES", fill=MINT, font=font(18, True))
    draw.text(
        (25, 50),
        "Same frozen perception model. Different connected interiors.",
        fill=TEXT,
        font=font(28, True),
    )
    for index, case in enumerate(cases):
        y = 100 + index * row_height
        draw.rounded_rectangle((15, y, width - 15, y + row_height - 12), radius=12, fill=PANEL)
        draw.text(
            (30, y + 12), case["title"], fill=COLORS[index % len(COLORS)], font=font(23, True)
        )
        directory = root / case["id"]
        experiment = read_json(directory / "run/experiment.json", {})
        records = experiment.get("trace", [])
        positions = (
            [round((len(records) - 1) * fraction) for fraction in (0.15, 0.42, 0.69, 0.95)]
            if records
            else []
        )
        sources = [directory / "capture/top_down.png"] + [
            Path(records[position]["overlay_path"]) for position in positions
        ]
        titles = ["Evaluator top-down"] + [
            f"Acquired overlay / {records[position]['id']}" for position in positions
        ]
        for column, (source, title) in enumerate(zip(sources, titles, strict=True)):
            x = 28 + column * 354
            label_color = MUTED if column == 0 else COLORS[column - 1]
            draw.text((x, y + 48), title, fill=label_color, font=font(13))
            if column:
                draw.rectangle((x - 2, y + 68, x + 342, y + 321), outline=label_color, width=2)
            if source.is_file():
                image = Image.open(source).convert("RGB")
                if column == 0 and records:
                    reference = read_json(directory / "capture/reference.json")
                    for position, color in zip(positions, COLORS, strict=True):
                        image = top_down_image(reference, image, records[position], [], color)
                image = ImageOps.contain(image, (340, 247))
                canvas.paste(
                    image, (x + (340 - image.width) // 2, y + 72 + (247 - image.height) // 2)
                )
        label = (
            "No completed evaluation"
            if not case["evaluated"]
            else f"{case['visits']['rooms_entered']}/{case['visits']['reference_room_count']} rooms entered  ·  {case['metrics']['visible_edge_coverage']:.1%} edge visibility  ·  {case['metrics']['edge_completeness_10cm']:.1%} 3D completeness at 10 cm"
        )
        draw.text((30, y + 321), label, fill=MUTED, font=font(14))
    canvas.save(output / "overview.jpg", quality=93)


def percent(value):
    return "—" if value is None else f"{value:.1%}"


def write_html(summary, root, output):
    rows, details = [], []
    for case in summary["cases"]:
        title = html.escape(case["title"])
        if not case["evaluated"]:
            rows.append(
                f"<tr><th>{title}</th><td colspan='7'>{html.escape(case['status'])} / no evaluation</td></tr>"
            )
            details.append(
                f"<article><h3>{title}</h3><p>{html.escape(case['description'])}</p><p>Case stage: {html.escape(str(case['stage']))}; status: {html.escape(case['status'])}. No completed evaluation is available.</p></article>"
            )
            continue
        visits, metrics, audit = case["visits"], case["metrics"], case["path_audit"]
        rows.append(
            f"<tr><th>{title}</th><td>{visits['rooms_entered']}/{visits['reference_room_count']}</td>"
            + "".join(f"<td>{percent(metrics[key])}</td>" for key in METRICS)
            + f"<td>{html.escape(case['stop_reason'])}<br>{html.escape(case['status'])} / {html.escape(str(case['stage']))}</td></tr>"
        )
        report = Path(os.path.relpath(root / case["id"] / "report/report.html", output)).as_posix()
        video = Path(os.path.relpath(root / case["id"] / "report/replay.mp4", output)).as_posix()
        poster = Path(os.path.relpath(root / case["id"] / "report/poster.jpg", output)).as_posix()
        entry = " → ".join(
            html.escape(item.replace("_", " ")) for item in visits["region_entry_order"]
        )
        architecture = audit["architecture"]["intersecting_path_segments"]
        furniture = audit["furniture_recipe_aabb_proxy"]["intersecting_path_segments"]
        full_seconds = case.get("full_pipeline_wall_seconds")
        full_timing = (
            f"Full pipeline including startup, evaluation and export: {full_seconds:.1f} s."
            if full_seconds is not None
            else "Full pipeline timing unavailable."
        )
        artifacts = case.get("available_artifacts", [])
        replay = (
            f"<video controls playsinline muted loop preload='metadata' poster='{poster}'><source src='{video}' type='video/mp4'></video>"
            if "replay.mp4" in artifacts
            else "<p>A completed replay is unavailable for this case.</p>"
        )
        report_link = (
            f"<p><a href='{report}'>Open interactive case report ↗</a></p>"
            if "report.html" in artifacts
            else ""
        )
        details.append(
            f"<article><h3>{title}</h3><p>{html.escape(case['description'])}</p>{replay}<p>{case['frames']} acquired images · {case['path_length_m']:.1f} m · {case['action_seconds']:.1f} simulated action seconds · {case['wall_seconds']:.1f} acquisition/processing seconds (excluding renderer startup and reporting). {full_timing}</p><p>First entries: {entry}. Paths inside observed safe map: {audit['all_motions_in_observed_safe_map']}. Architecture/furniture proxy intersections: {architecture}/{furniture}. Case status: {html.escape(case['status'])}, stage: {html.escape(str(case['stage']))}.</p>{report_link}</article>"
        )
    aggregates = summary["aggregate"]
    table = "".join(
        f"<tr><th>{html.escape(key.replace('_', ' '))}</th><td>{percent(aggregates['macro'][key]['value'])}</td><td>{aggregates['macro'][key]['denominator_buildings']} buildings</td><td>{percent(aggregates['weighted'][key]['value'])}</td><td>{aggregates['weighted'][key]['denominator_value']:,.1f} {html.escape(aggregates['weighted'][key]['denominator'].replace('_', ' '))}</td></tr>"
        for key in METRICS
    )
    page = (
        TEMPLATE.replace("__CASE_ROWS__", "".join(rows))
        .replace("__DETAILS__", "".join(details))
        .replace("__AGGREGATE_ROWS__", table)
        .replace(
            "__COUNTS__",
            f"{aggregates['evaluated_buildings']}/{aggregates['declared_buildings']} declared buildings evaluated",
        )
    )
    (output / "report.html").write_text(page)


TEMPLATE = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RoomGraph — Multiple connected spaces</title><style>body{margin:0;background:#0c1420;color:#e7f0f7;font:16px system-ui}main{max-width:1440px;margin:auto;padding:24px}h1{font-size:clamp(32px,5vw,60px)}p{line-height:1.65;color:#a9bed0;max-width:100ch}a{color:#55e4bd}img,video{width:100%;border-radius:14px}section{margin-top:50px}.scroll{overflow:auto}table{border-collapse:collapse;min-width:800px;width:100%;font-size:14px}th,td{padding:12px;text-align:left;border-bottom:1px solid #33465a}.cases{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px}article{background:#142131;border-radius:16px;padding:20px}.note{border-left:3px solid #efc187;padding:16px;background:#142131}@media(max-width:750px){.cases{grid-template-columns:1fr}main{padding:14px}}</style><main><p>ROOMGRAPH / MULTIPLE SPACE CASES</p><h1>Explore different spaces.<br>Keep the missing structure visible.</h1><p>Four furnished development buildings exercise branches, loops, open workspaces and compact clutter. The same frozen RGB edge model and frontier policy acquire ideal RGB-D with known poses. The planner receives acquired observations, not a floor plan or reference room labels.</p><p class="note">__COUNTS__. These are development cases, not a held-out robustness benchmark. Entering every reference room does not establish complete reconstruction. Simulation motion is kinematic; it is not a walking controller.</p><img src="overview.jpg" alt="Each case has a furnished reference top-down and four acquired learned-edge overlays"><section><h2>Visibility and geometry, measured separately.</h2><div class="scroll"><table><thead><tr><th>Building</th><th>Rooms entered</th><th>Visibility</th><th>3D precision</th><th>3D completeness</th><th>Typed precision</th><th>Typed completeness</th><th>Stop reason</th></tr></thead><tbody>__CASE_ROWS__</tbody></table></div><p>All 3D metrics use 10 cm tolerance. Visibility is true edge length seen by acquired cameras; completeness is reference edge length reconstructed within tolerance. Precision uses fused predicted edge voxels. Typed scores also require the correct structural channel.</p><img src="comparison.png" alt="Per-case visibility, geometric and typed edge reconstruction metrics"><img src="coverage.png" alt="Reference architectural edge visibility versus body travel and simulated action time"></section><section><h2>Inspect every declared case.</h2><div class="cases">__DETAILS__</div></section><section><h2>Aggregate with explicit denominators.</h2><div class="scroll"><table><thead><tr><th>Measurement</th><th>Per-building macro</th><th>Macro denominator</th><th>Pooled weighted</th><th>Weighted denominator</th></tr></thead><tbody>__AGGREGATE_ROWS__</tbody></table></div><p>Unavailable evaluations stay visible as missing rows and are excluded from measured aggregates. Every aggregate records its denominator; no unmeasured success or zero is invented. Reference lengths include all architectural edges, including unvisited regions. Furniture collision checks use recipe bounding boxes and do not validate whole-body motion.</p><p><a href="summary.json">Measured summary and frozen provenance</a></p></section></main></html>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    suite, frozen_sha256 = frozen_suite(args.suite, args.root)
    cases = [load_case(spec, args.root) for spec in suite["cases"]]
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Case IDs must be unique")
    checkpoints = {case["provenance"]["checkpoint_sha256"] for case in cases if case["evaluated"]}
    if len(checkpoints) > 1:
        raise ValueError("Suite cases do not share the frozen perception checkpoint")
    summary = {
        "schema_version": 1,
        "name": suite["name"],
        "status": "complete"
        if all(case["status"] in ("complete", "failed", "error") for case in cases)
        else "partial",
        "suite_sha256": sha256(args.suite),
        "frozen_suite_sha256": frozen_sha256,
        "shared_budget": suite["shared_budget"],
        "aggregate": aggregate(cases),
        "cases": cases,
        "limitations": [
            "Four development buildings; no held-out generalization or policy comparison",
            "Ideal simulated depth and known metric poses; kinematic humanoid proxy",
            "Partial observed surface/edge maps; no completed watertight building mesh",
            "Physically open portals; no door-leaf articulation or opening-angle inference",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    make_plots(cases, args.output)
    make_contact_sheet(cases, args.root, args.output)
    write_html(summary, args.root, args.output)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {"report": str(args.output / "report.html"), "aggregate": summary["aggregate"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
