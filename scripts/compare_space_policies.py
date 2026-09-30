"""Compare frozen exploration policies on matching development buildings and budgets."""

# Long HTML lines preserve readable markup.
# ruff: noqa: E501

import argparse
import hashlib
import html
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np

METRICS = (
    "rooms_entered",
    "visible_edge_coverage",
    "edge_completeness_10cm",
    "edge_precision_10cm",
    "typed_edge_completeness_10cm",
    "typed_edge_precision_10cm",
    "path_length_m",
    "action_seconds",
    "wall_seconds",
    "full_pipeline_wall_seconds",
    "frames",
)
POLICY_CONFIG_FIELDS = {"name", "exploration_policy"}
BG, PANEL, TEXT, MUTED = "#0c1420", "#142131", "#e7f0f7", "#a9bed0"
AMBER, MINT = "#efc187", "#55e4bd"
DIAGNOSTIC_PROTOCOL = {
    "low_surface_gain_threshold_voxels": 500,
    "revisited_pose_radius_m": 0.3,
    "revisited_pose_excludes_immediately_previous_station": True,
    "goal_switch_remaining_distance_threshold_m": 0.75,
    "goal_switch_target_displacement_threshold_m": 1.0,
    "goal_switch_qualification": (
        "Counts target changes over 1 m while the previous target remains over 0.75 m away. "
        "A switch can be necessary when observations invalidate a path or resolve a frontier."
    ),
    "qualification": (
        "Low surface gain and nearby repeated poses are descriptive proxies, not proof that "
        "images are redundant: they may improve confidence, edge predictions or navigation."
    ),
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def index_cases(cases):
    indexed = {}
    for case in cases:
        identifier = case["id"]
        if not re.fullmatch(r"[a-z][a-z0-9_]*", identifier) or identifier in indexed:
            raise ValueError("Case IDs must be unique safe identifiers")
        indexed[identifier] = case
    return indexed


def diagnostics(experiment):
    """Measure actual repeated observations without access to reference geometry."""
    stations = experiment.get("stations", [])
    result = {
        "low_surface_gain_stations": 0,
        "low_surface_gain_frames": 0,
        "revisited_pose_stations": 0,
        "revisited_pose_frames": 0,
        "exact_revisited_pose_stations": 0,
        "exact_revisited_pose_frames": 0,
        "goal_switches_before_proximity": 0,
    }
    for index, station in enumerate(stations):
        frames = len(station["view_ids"])
        if station["new_surface_voxels"] < 500:
            result["low_surface_gain_stations"] += 1
            result["low_surface_gain_frames"] += frames
        if index > 1:
            previous = np.array([s["position"][:2] for s in stations[: index - 1]])
            distances = np.linalg.norm(previous - station["position"][:2], axis=1)
            if distances.min() < 0.3:
                result["revisited_pose_stations"] += 1
                result["revisited_pose_frames"] += frames
            if distances.min() < 1e-6:
                result["exact_revisited_pose_stations"] += 1
                result["exact_revisited_pose_frames"] += frames
        if index:
            previous_goal = stations[index - 1].get("selected_goal")
            goal = station.get("selected_goal")
            if previous_goal is not None and goal is not None:
                remaining = np.linalg.norm(np.array(previous_goal) - station["position"][:2])
                change = np.linalg.norm(np.array(previous_goal) - goal)
                if remaining > 0.75 and change > 1.0:
                    result["goal_switches_before_proximity"] += 1
    return result


def verify_receipt(case, directory):
    receipt = directory / "receipt.json"
    if case.get("receipt_sha256"):
        if sha256(receipt) != case["receipt_sha256"]:
            raise ValueError(f"Changed receipt: {case['id']}")
        for name, digest in read_json(receipt)["output_sha256"].items():
            artifact = (directory / name).resolve()
            if not artifact.is_relative_to(directory.resolve()) or sha256(artifact) != digest:
                raise ValueError(f"Changed receipted artifact: {case['id']}/{name}")
    elif case["status"] == "complete":
        raise ValueError(f"Completed case lacks receipt: {case['id']}")


def validate_summary_measurements(case, evaluation, reference, experiment):
    """Reject stale plotted curves and outcome labels as well as final scores."""
    identifier = case["id"]
    metrics = {key: evaluation["final"][key] for key in METRICS if "edge" in key}
    if case["metrics"] != metrics:
        raise ValueError(f"Summary metrics differ from evaluation: {identifier}")
    visit_fields = (
        "rooms_entered",
        "reference_room_count",
        "region_entry_order",
        "region_transition_sequence",
        "unique_portals_crossed",
    )
    for key in visit_fields:
        if case["visits"].get(key) != evaluation["visits"].get(key):
            raise ValueError(f"Summary visit {key} differs from evaluation: {identifier}")
    missing = [
        room["id"]
        for room in reference["rooms"]
        if room["kind"] == "room" and room["id"] not in evaluation["visits"]["region_entry_order"]
    ]
    if case["unentered_reference_rooms"] != missing:
        raise ValueError(f"Summary missing rooms differ from reference visits: {identifier}")
    if case["visibility_curve"] != evaluation["visibility_curve"]:
        raise ValueError(f"Summary visibility curve differs from evaluation: {identifier}")
    if case["stop_reason"] != evaluation["stop_reason"]:
        raise ValueError(f"Summary stop reason differs from evaluation: {identifier}")
    if case["frames"] != len(experiment["trace"]):
        raise ValueError(f"Summary frame count differs from experiment: {identifier}")
    for key in ("wall_seconds", "action_seconds", "path_length_m"):
        if case[key] != experiment[key]:
            raise ValueError(f"Summary {key} differs from experiment: {identifier}")


def load_side(report, root):
    """Verify summaries against completed experiments and their frozen receipts."""
    summary = read_json(report / "summary.json")
    if summary["status"] != "complete":
        raise ValueError("Comparison requires terminal suites, retaining failed cases")
    frozen = read_json(root / "suite.json")
    if sha256(root / "suite.json") != summary["frozen_suite_sha256"]:
        raise ValueError("Suite summary and frozen inputs differ")
    cases = index_cases(summary["cases"])
    declared = index_cases(frozen["suite"]["cases"])
    if cases.keys() != declared.keys():
        raise ValueError("Summary dropped or added declared cases")
    if summary["shared_budget"] != frozen["suite"]["shared_budget"]:
        raise ValueError("Summary budget differs from frozen suite")
    experiments, manifests = {}, {}
    for identifier, case in cases.items():
        directory = root / identifier
        verify_receipt(case, directory)
        experiment_path = directory / "run/experiment.json"
        experiment = read_json(experiment_path) if experiment_path.exists() else None
        manifest_path = directory / "capture/manifest.json"
        manifest = read_json(manifest_path) if manifest_path.exists() else None
        experiments[identifier], manifests[identifier] = experiment, manifest
        if experiment:
            if experiment["config"] != frozen["case_configs"][identifier]:
                raise ValueError(f"Experiment config differs from frozen suite: {identifier}")
            checkpoint = frozen["input_sha256"][frozen["suite"]["checkpoint"]]
            if experiment["checkpoint_sha256"] != checkpoint:
                raise ValueError(f"Checkpoint differs from frozen suite: {identifier}")
        if manifest:
            layout = frozen["input_sha256"][declared[identifier]["building"]]
            if manifest["source_config_sha256"] != layout:
                raise ValueError(f"Captured layout differs from frozen input: {identifier}")
        if not case["evaluated"]:
            evaluation_path = directory / "run/evaluation.json"
            if evaluation_path.exists() and read_json(evaluation_path).get("status") == "complete":
                raise ValueError(f"Summary omits an available evaluation: {identifier}")
            continue
        if not experiment or experiment.get("status") != "complete":
            raise ValueError(f"Measured case lacks completed experiment: {identifier}")
        evaluation_path = directory / "run/evaluation.json"
        evaluation = read_json(evaluation_path)
        reference_path = directory / "capture/reference.json"
        provenance = case["provenance"]
        for field, path in (
            ("experiment_sha256", experiment_path),
            ("evaluation_sha256", evaluation_path),
            ("reference_sha256", reference_path),
        ):
            if provenance[field] != sha256(path):
                raise ValueError(f"Stale {field}: {identifier}")
        if evaluation["provenance"]["experiment_sha256"] != sha256(experiment_path):
            raise ValueError(f"Evaluation uses another experiment: {identifier}")
        if evaluation["provenance"]["reference_sha256"] != sha256(reference_path):
            raise ValueError(f"Evaluation uses another reference: {identifier}")
        validate_summary_measurements(case, evaluation, read_json(reference_path), experiment)
        replay_path = directory / "run/replay_check.json"
        replay = read_json(replay_path) if replay_path.exists() else {}
        if replay.get("status") == "pass" and replay.get("experiment_sha256") != sha256(
            experiment_path
        ):
            raise ValueError(f"Replay references another experiment: {identifier}")
        case["_replay_qualified"] = replay.get("status") == "pass"
    return {
        "summary": summary,
        "frozen": frozen,
        "cases": cases,
        "experiments": experiments,
        "manifests": manifests,
        "summary_sha256": sha256(report / "summary.json"),
    }


def validate_pair(baseline, candidate):
    """Allow only declared policy/name differences; reject changed layout or sensors."""
    if baseline["cases"].keys() != candidate["cases"].keys():
        raise ValueError("Both policies must retain exactly the same declared cases")
    bf, cf = baseline["frozen"], candidate["frozen"]
    if bf["suite"]["shared_budget"] != cf["suite"]["shared_budget"]:
        raise ValueError("Policy comparison requires identical shared budgets")
    bp = bf["input_sha256"][bf["suite"]["checkpoint"]]
    cp = cf["input_sha256"][cf["suite"]["checkpoint"]]
    if bp != cp:
        raise ValueError("Policy comparison requires the same frozen checkpoint")
    bs, cs = index_cases(bf["suite"]["cases"]), index_cases(cf["suite"]["cases"])
    policies = [e.get("exploration_policy") for e in candidate["experiments"].values() if e]
    if any(not p or p.get("id") != "frontier_commitment_v2" for p in policies):
        raise ValueError("Candidate must explicitly declare frontier_commitment_v2")
    if policies and any(p != policies[0] for p in policies):
        raise ValueError("Candidate policy settings must remain fixed across cases")
    declared_policy = cf["suite"].get("exploration_policy")
    if not declared_policy or any(p != declared_policy for p in policies):
        raise ValueError("Candidate policy differs from the frozen suite declaration")
    for identifier in baseline["cases"]:
        if (
            bf["input_sha256"][bs[identifier]["building"]]
            != cf["input_sha256"][cs[identifier]["building"]]
        ):
            raise ValueError(f"Layout mismatch: {identifier}")
        bc, cc = bf["case_configs"][identifier], cf["case_configs"][identifier]
        b_settings = {k: v for k, v in bc.items() if k not in POLICY_CONFIG_FIELDS}
        c_settings = {k: v for k, v in cc.items() if k not in POLICY_CONFIG_FIELDS}
        if b_settings != c_settings:
            raise ValueError(f"Experiment settings or start mismatch: {identifier}")
        if bs[identifier].get("split") != cs[identifier].get("split"):
            raise ValueError(f"Case split mismatch: {identifier}")
        bm, cm = baseline["manifests"][identifier], candidate["manifests"][identifier]
        if bm and cm:
            for key in ("renderer", "sensor", "samples_per_pixel"):
                if bm[key] != cm[key]:
                    raise ValueError(f"Capture {key} mismatch: {identifier}")
    # Existing runtime, sensor, renderer, learner and evaluator code stays byte-identical.
    shared = bf["input_sha256"].keys() & cf["input_sha256"].keys()
    for name in shared:
        if (name.startswith("src/") or name.startswith("scripts/")) and (
            bf["input_sha256"][name] != cf["input_sha256"][name]
        ):
            raise ValueError(f"Shared implementation changed: {name}")


def case_measurements(case, experiment):
    result = {
        key: case.get(key)
        for key in (
            "status",
            "evaluated",
            "stop_reason",
            "unentered_reference_rooms",
            "frames",
            "path_length_m",
            "action_seconds",
            "wall_seconds",
            "full_pipeline_wall_seconds",
            "available_artifacts",
            "media_sha256",
            "receipt_sha256",
            "provenance",
        )
    }
    result.update({key: case.get("metrics", {}).get(key) for key in METRICS if "edge" in key})
    result["source_evaluated"] = case["evaluated"]
    result["replay_qualified"] = case.get("_replay_qualified", case["evaluated"])
    result["evaluated"] = case["evaluated"] and result["replay_qualified"]
    result["availability_reason"] = (
        "missing_or_failed_replay"
        if case["evaluated"] and not result["replay_qualified"]
        else None
        if case["evaluated"]
        else "no_completed_evaluation"
    )
    result["rooms_entered"] = case.get("visits", {}).get("rooms_entered")
    result["reference_room_count"] = case.get("visits", {}).get("reference_room_count")
    if not result["evaluated"]:
        for key in METRICS:
            if key == "rooms_entered" or "edge" in key:
                result[key] = None
        result["unentered_reference_rooms"] = None
    result["diagnostics"] = diagnostics(experiment) if experiment else None
    result["policy"] = experiment.get("exploration_policy") if experiment else None
    decisions = experiment.get("policy_decisions", []) if experiment else []
    if decisions:
        if len(decisions) != len(experiment["stations"]):
            raise ValueError("Policy decisions must match acquired survey stations")
        result["policy_diagnostics"] = {
            "decision_count": len(decisions),
            "retained_different_from_local_greedy": sum(
                d["action"] == "retained" and d.get("selected_goal_xy") != d.get("baseline_goal_xy")
                for d in decisions
            ),
            "local_greedy_qualification": (
                "Greedy choices are evaluated on candidate observations at that station; "
                "they are not counterfactual outcomes of the complete baseline run."
            ),
            "action_counts": dict(Counter(d["action"] for d in decisions)),
            "release_reason_counts": dict(
                Counter(d["release_reason"] for d in decisions if d.get("release_reason"))
            ),
            "commitment_episodes": len(
                {d["commitment_id"] for d in decisions if d.get("commitment_id") is not None}
            ),
        }
    else:
        result["policy_diagnostics"] = None
    return result


def finite(value):
    return isinstance(value, (float, int)) and math.isfinite(value)


def build_comparison(baseline, candidate):
    validate_pair(baseline, candidate)
    pairs = []
    for identifier, case in baseline["cases"].items():
        b = case_measurements(case, baseline["experiments"][identifier])
        c = case_measurements(candidate["cases"][identifier], candidate["experiments"][identifier])
        pairs.append(
            {
                "id": identifier,
                "title": case["title"],
                "split": case.get("split", "development"),
                "baseline": b,
                "candidate": c,
                "delta": {
                    key: c[key] - b[key]
                    if b["evaluated"] and c["evaluated"] and finite(c[key]) and finite(b[key])
                    else None
                    for key in METRICS
                },
            }
        )
    macro = {}
    for key in METRICS:
        selected = [
            p
            for p in pairs
            if p["baseline"]["evaluated"]
            and p["candidate"]["evaluated"]
            and p["delta"][key] is not None
        ]
        b = float(np.mean([p["baseline"][key] for p in selected])) if selected else None
        c = float(np.mean([p["candidate"][key] for p in selected])) if selected else None
        macro[key] = {
            "baseline": b,
            "candidate": c,
            "delta": c - b if selected else None,
            "denominator_buildings": len(selected),
            "case_ids": [p["id"] for p in selected],
        }
    return {
        "schema_version": 1,
        "comparison": "greedy_frontiers_v1_vs_goal_commitment_v2",
        "status": "complete",
        "qualification": "Tuned development comparison on the same four layouts; not held-out generalization.",
        "provenance": {
            "baseline_summary_sha256": baseline["summary_sha256"],
            "candidate_summary_sha256": candidate["summary_sha256"],
            "baseline_frozen_suite_sha256": baseline["summary"]["frozen_suite_sha256"],
            "candidate_frozen_suite_sha256": candidate["summary"]["frozen_suite_sha256"],
        },
        "shared_budget": baseline["frozen"]["suite"]["shared_budget"],
        "diagnostic_protocol": DIAGNOSTIC_PROTOCOL,
        "cases": pairs,
        "macro": macro,
        "limitations": [
            "Ideal simulated RGB-D and known poses; original approximate humanoid proxy.",
            "One acquisition run per policy and layout; differences are not significance estimates.",
            "Fresh acquisitions use matching rendering settings but are not identical-RGB "
            "counterfactuals; renderer sampling can vary between runs.",
            "Reference room names and geometry are evaluator-only inputs.",
            "All paired regressions and missing evaluations remain visible; cases lacking a "
            "successful matching replay are unavailable for paired metrics.",
            "Acquisition/processing wall time excludes renderer startup and report generation.",
            "Geometric reconstruction remains partial; room entry is not full reconstruction.",
        ],
    }


def make_plots(comparison, baseline, candidate, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = {
        "figure.facecolor": BG,
        "axes.facecolor": PANEL,
        "axes.edgecolor": MUTED,
        "text.color": TEXT,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": TEXT,
        "font.size": 11,
        "savefig.facecolor": BG,
    }
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 2, figsize=(15, 9), layout="constrained")
        for ax, metric, title in zip(
            axes.flat,
            (
                "rooms_entered",
                "visible_edge_coverage",
                "edge_completeness_10cm",
                "edge_precision_10cm",
            ),
            (
                "Rooms entered",
                "Architectural edge visibility",
                "3D completeness @ 10 cm",
                "3D precision @ 10 cm",
            ),
            strict=True,
        ):
            factor = 1 if metric == "rooms_entered" else 100
            for index, pair in enumerate(comparison["cases"]):
                b, c = pair["baseline"][metric], pair["candidate"][metric]
                if finite(b) and finite(c):
                    ax.plot([b * factor, c * factor], [index, index], color=MUTED, lw=2)
                for value, color, marker in ((b, AMBER, "o"), (c, MINT, "D")):
                    if finite(value):
                        ax.scatter(
                            value * factor, index, s=75, color=color, marker=marker, zorder=3
                        )
                delta = pair["delta"][metric]
                label = "unavailable" if delta is None else f"{delta * factor:+.1f}"
                ax.text(
                    0.98,
                    index + 0.19,
                    label,
                    transform=ax.get_yaxis_transform(),
                    ha="right",
                    fontsize=10,
                    color=TEXT,
                )
            ax.set_yticks(
                range(len(comparison["cases"])),
                [p["id"].replace("_", " ").title() for p in comparison["cases"]],
            )
            ax.invert_yaxis()
            ax.set_ylim(len(comparison["cases"]) - 0.5, -0.6)
            ax.set_title(title, loc="left", pad=16, weight="bold")
            ax.set_xlabel(
                "Rooms · annotations are changes"
                if factor == 1
                else "Percent · annotations are percentage-point changes"
            )
            ax.set_xlim((0, 4.5) if factor == 1 else (0, 105))
            ax.grid(axis="x", alpha=0.15)
        axes.flat[0].scatter([], [], color=AMBER, label="v1 · greedy", s=75)
        axes.flat[0].scatter([], [], color=MINT, marker="D", label="v2 · commitment", s=75)
        axes.flat[0].legend(facecolor=PANEL, labelcolor=TEXT, loc="lower left", framealpha=0.95)
        fig.suptitle("Same spaces. Same model. A different frontier decision.", fontsize=22)
        fig.savefig(output / "paired_metrics.png", dpi=150)
        plt.close(fig)
        fig, axes = plt.subplots(2, 2, figsize=(15, 8), layout="constrained")
        for ax, pair in zip(axes.flat, comparison["cases"], strict=True):
            for bundle, color, label in ((baseline, AMBER, "v1"), (candidate, MINT, "v2")):
                side = "baseline" if label == "v1" else "candidate"
                if not pair[side]["evaluated"]:
                    ax.text(
                        0.02,
                        0.95 if label == "v1" else 0.85,
                        f"{label}: verified curve unavailable",
                        transform=ax.transAxes,
                        color=color,
                    )
                    continue
                curve = bundle["cases"][pair["id"]].get("visibility_curve", [])
                if curve:
                    ax.plot(
                        [p["action_seconds"] for p in curve],
                        [100 * p["visible_edge_coverage"] for p in curve],
                        color=color,
                        label=label,
                        lw=2,
                    )
            ax.set_title(pair["id"].replace("_", " ").title(), loc="left", weight="bold")
            ax.set(
                xlabel="Simulated action seconds",
                ylabel="Visible reference edge length (%)",
                ylim=(0, 100),
            )
            ax.grid(alpha=0.12)
            ax.legend(facecolor=PANEL, labelcolor=TEXT)
        fig.suptitle("Coverage as the head camera observes and the body moves", fontsize=21)
        fig.savefig(output / "visibility_curves.png", dpi=150)
        plt.close(fig)


def formatted(value, key, signed=False):
    if value is None:
        return "—"
    sign = "+" if signed else ""
    if "edge" in key:
        return f"{value * 100:{sign}.1f}" + (" pp" if signed else "%")
    return f"{value:{sign}.1f}"


def comparison_rows(comparison, keys):
    rows = []
    for pair in comparison["cases"]:
        cells = []
        for key in keys:
            b, c, d = pair["baseline"][key], pair["candidate"][key], pair["delta"][key]
            cells.append(
                f"<td>{formatted(b, key)} → {formatted(c, key)}"
                f"<small>{formatted(d, key, True)}</small></td>"
            )
        rows.append(
            f"<tr><th>{html.escape(pair['id'].replace('_', ' ').title())}</th>"
            + "".join(cells)
            + "</tr>"
        )
    cells = []
    for key in keys:
        metric = comparison["macro"][key]
        cells.append(
            f"<td>{formatted(metric['baseline'], key)} → "
            f"{formatted(metric['candidate'], key)}"
            f"<small>{formatted(metric['delta'], key, True)} · "
            f"n={metric['denominator_buildings']}</small></td>"
        )
    rows.append("<tr><th>Paired macro</th>" + "".join(cells) + "</tr>")
    return "".join(rows)


def write_html(comparison, baseline_root, candidate_root, output):
    cards, diagnostic_rows = [], []
    for pair in comparison["cases"]:
        cells = []
        for key in (
            "low_surface_gain_frames",
            "revisited_pose_frames",
            "goal_switches_before_proximity",
        ):
            before = (pair["baseline"].get("diagnostics") or {}).get(key)
            after = (pair["candidate"].get("diagnostics") or {}).get(key)
            cells.append(
                f"<td>{before if before is not None else '—'} → "
                f"{after if after is not None else '—'}</td>"
            )
        diagnostic_rows.append(
            f"<tr><th>{html.escape(pair['id'].replace('_', ' ').title())}</th>"
            + "".join(cells)
            + "</tr>"
        )
        videos = []
        for side, root in (("baseline", baseline_root), ("candidate", candidate_root)):
            run = pair[side]
            prefix = Path(os.path.relpath(root / pair["id"] / "report", output)).as_posix()
            missing_rooms = run.get("unentered_reference_rooms")
            missing = "unavailable" if missing_rooms is None else ", ".join(missing_rooms) or "none"
            video = (
                (
                    f'<video controls muted playsinline preload="metadata" '
                    f'poster="{prefix}/poster.jpg"><source src="{prefix}/replay.mp4" '
                    'type="video/mp4"></video>'
                )
                if "replay.mp4" in (run.get("available_artifacts") or [])
                else "<p>Replay unavailable.</p>"
            )
            videos.append(
                f"<article><h3>{side.title()}</h3>{video}<p>Unentered rooms: "
                f"{html.escape(missing)}. Status: {html.escape(run['status'])}.</p>"
                f'<a href="{prefix}/report.html">Full acquired report ↗</a></article>'
            )
        cards.append(
            f"<h3>{html.escape(pair['title'])}</h3><div class='cards'>" + "".join(videos) + "</div>"
        )
    main_keys = (
        "rooms_entered",
        "visible_edge_coverage",
        "edge_completeness_10cm",
        "edge_precision_10cm",
    )
    efficiency_keys = ("path_length_m", "action_seconds", "wall_seconds", "frames")
    missed = [
        f"{pair['id'].replace('_', ' ')} ({pair['candidate']['rooms_entered']}/{pair['candidate']['reference_room_count']} rooms)"
        for pair in comparison["cases"]
        if pair["candidate"]["evaluated"]
        and pair["candidate"]["rooms_entered"] < pair["candidate"]["reference_room_count"]
    ]
    outcome = (
        "Among replay-qualified candidate cases, missed rooms: "
        + ("; ".join(missed) if missed else "none")
        + ". "
    )
    outcome += (
        "Macro completeness change: "
        + formatted(
            comparison["macro"]["edge_completeness_10cm"]["delta"], "edge_completeness_10cm", True
        )
        + ". The original v1 policy remains the default; v2 is an experimental candidate."
    )
    document = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>RoomGraph policy comparison</title>
<style>body{{margin:0;background:{BG};color:{TEXT};font:16px system-ui}}main{{max-width:1450px;margin:auto;padding:28px}}
h1{{font-size:clamp(34px,5vw,64px)}}p{{color:{MUTED};line-height:1.65}}a{{color:{MINT}}}img,video{{width:100%;border-radius:14px}}
section{{margin-top:50px}}.scroll{{overflow:auto}}table{{width:100%;min-width:820px;border-collapse:collapse}}
th,td{{padding:15px;text-align:left;border-bottom:1px solid #33465a}}small{{display:block;color:{MUTED};margin-top:5px}}
.cards{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}}article{{background:{PANEL};padding:20px;border-radius:16px}}
.note{{border-left:3px solid {AMBER};padding:15px;background:{PANEL}}}@media(max-width:700px){{.cards{{grid-template-columns:1fr}}}}</style>
<main><p>ROOMGRAPH / EXPLORATION POLICY COMPARISON</p><h1>Finish the approach.<br>Measure what changes.</h1>
<p class="note"><strong>{html.escape(outcome)}</strong><br>{html.escape(comparison["qualification"])} All cases and regressions remain visible. Ideal RGB-D and known poses assist mapping; reconstruction stays partial.</p>
<p>v1 reselects a greedy frontier after every local sweep. v2 persists a frontier approach while checking current observed-free paths and stopping conditions. The layouts, starts, rendering, model, scans and per-case budgets stay fixed. The policy sees no reference room names or geometry.</p>
<img src="paired_metrics.png" alt="Paired per-building room entry, architectural visibility, completeness and precision">
<section><h2>Paired measured outcomes</h2><div class="scroll"><table><thead><tr><th>Case</th><th>Rooms</th><th>Visibility</th><th>3D completeness</th><th>3D precision</th></tr></thead><tbody>{comparison_rows(comparison, main_keys)}</tbody></table></div>
<p>Each cell shows v1 → v2 and its signed change. Geometric scores use 10 cm tolerance. Macro averages use only measured pairs and state their denominators; missing cases remain explicit.</p></section>
<section><h2>Travel and time</h2><div class="scroll"><table><thead><tr><th>Case</th><th>Travel (m)</th><th>Action (s)</th><th>Acquisition / processing (s)</th><th>Frames</th></tr></thead><tbody>{comparison_rows(comparison, efficiency_keys)}</tbody></table></div>
<p>Action time is simulated body/head motion and settling. Acquisition/processing time is measured wall time, excluding renderer startup, evaluation and media export. One run per policy and case does not establish a statistically reliable speedup. Fresh acquisitions share rendering settings but may have small RGB sampling differences, even at matching poses.</p><img src="visibility_curves.png" alt="Visibility versus simulated action time for both policies"></section>
<section><h2>Acquired replays, side by side</h2>{"".join(cards)}</section>
<section><h2>Observation diagnostics</h2><div class="scroll"><table><thead><tr><th>Case</th><th>Low surface gain frames</th><th>Frames near earlier poses</th><th>Early goal switches</th></tr></thead><tbody>{"".join(diagnostic_rows)}</tbody></table></div><p>{html.escape(DIAGNOSTIC_PROTOCOL["qualification"])} Low gain means fewer than 500 new surface voxels per survey station. A repeated pose is within 0.3 m of an earlier nonadjacent station. Early goal switches change the target by more than 1 m while the previous target remains more than 0.75 m away; a switch may be required by newly observed obstacles or resolved frontiers. Full policy metadata and frozen provenance are in the JSON.</p><a href="comparison.json">Measured comparison and diagnostics ↗</a></section></main></html>"""
    (output / "report.html").write_text(document)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "baseline-report",
        "baseline-root",
        "candidate-report",
        "candidate-root",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Comparison output must be fresh; retain previous comparisons")
    baseline = load_side(args.baseline_report, args.baseline_root)
    candidate = load_side(args.candidate_report, args.candidate_root)
    comparison = build_comparison(baseline, candidate)
    args.output.mkdir(parents=True)
    make_plots(comparison, baseline, candidate, args.output)
    write_html(comparison, args.baseline_root, args.candidate_root, args.output)
    comparison["media_sha256"] = {
        name: sha256(args.output / name) for name in ("paired_metrics.png", "visibility_curves.png")
    }
    (args.output / "comparison.json").write_text(
        json.dumps(comparison, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({"report": str(args.output / "report.html"), "macro": comparison["macro"]}))


if __name__ == "__main__":
    main()
