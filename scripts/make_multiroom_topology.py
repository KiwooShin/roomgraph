"""Visualize connectivity using only observed occupancy and inferred topology.

Run after exploration:
    PYTHONPATH=src python scripts/make_multiroom_topology.py \
        --run vis/multiroom/run_v4 --output vis/multiroom/report

Writes topology.png and a compact topology.json; no renderer reference is read.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

BG = "#0c1420"
PANEL = "#142131"
TEXT = "#e7f0f7"
MUTED = "#9bb0c5"
OCCUPIED = "#8997a8"
FRAGMENT = "#53677e"
COLORS = ("#55e4bd", "#6ba4f8", "#bc94f5", "#f0a4b5", "#ffbe70", "#89d4e6")


def compact_topology(topology, minimum_area_m2):
    """Keep original identities and all fragment counts while omitting the raster."""
    substantial, fragments = [], []
    for region in topology["regions"]:
        (substantial if region["observed_free_area_m2"] >= minimum_area_m2 else fragments).append(
            region
        )
    substantial.sort(
        key=lambda region: (
            region["first_observation_index"]
            if region["first_observation_index"] is not None
            else np.inf,
            region["id"],
        )
    )
    labels = {region["id"]: f"R{index + 1}" for index, region in enumerate(substantial)}
    regions = [
        {
            "label": labels[region["id"]],
            "source_id": region["id"],
            "centroid_xy_m": region["centroid_xy_m"],
            "observed_free_area_m2": region["observed_free_area_m2"],
            "touches_unknown": region["touches_unknown"],
            "reconstruction_complete": region["reconstruction_complete"],
        }
        for region in substantial
    ]
    connections = [
        {
            "source_id": portal["id"],
            "regions": [labels[identifier] for identifier in portal["regions"]],
            "center_xy_m": portal["center_xy_m"],
            "clearance_width_m": portal["clearance_width_m"],
            "verified_doorway": portal["verified_doorway"],
        }
        for portal in topology["portals"]
        if all(identifier in labels for identifier in portal["regions"])
    ]
    return {
        "schema_version": "0.1",
        "source": topology["source"],
        "method": topology["method"],
        "display_minimum_area_m2": minimum_area_m2,
        "substantial_region_count": len(regions),
        "neck_connection_count": len(connections),
        "small_fragment_count": len(fragments),
        "small_fragment_area_m2": sum(r["observed_free_area_m2"] for r in fragments),
        "regions": regions,
        "connections": connections,
        "small_fragments": [
            {"source_id": region["id"], "observed_free_area_m2": region["observed_free_area_m2"]}
            for region in fragments
        ],
        "limitations": topology["limitations"],
    }


def draw_topology(topology, states, summary, output):
    labels = np.asarray(topology["region_labels"])
    origin = np.asarray(topology["origin_xy_m"])
    resolution = float(topology["resolution_m"])
    if labels.shape != states.shape or np.any((labels > 0) & (states != 0)):
        raise ValueError("Topology labels must correspond to the final observed free-space map")
    indices = np.argwhere(states != -1)
    if not len(indices):
        raise ValueError("No observed occupancy cells to visualize")
    low = np.maximum(0, indices.min(axis=0) - 5)
    high = np.minimum(states.shape, indices.max(axis=0) + 6)
    image = np.full((*states.shape, 3), to_rgb(PANEL))
    image[states == 1] = to_rgb(OCCUPIED)
    image[states == 0] = to_rgb(FRAGMENT)
    palette = {}
    for index, region in enumerate(summary["regions"]):
        color = COLORS[index % len(COLORS)]
        palette[region["label"]] = color
        source_index = int(region["source_id"].split("_")[-1])
        image[labels == source_index] = to_rgb(color)
    extent = [
        origin[0] + low[1] * resolution,
        origin[0] + high[1] * resolution,
        origin[1] + low[0] * resolution,
        origin[1] + high[0] * resolution,
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 6.7), dpi=150, facecolor=BG)
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.22, top=0.79, wspace=0.13)
    count, necks = summary["substantial_region_count"], summary["neck_connection_count"]
    fig.text(0.04, 0.94, "Connectivity from acquired depth", color=TEXT, fontsize=23, weight="bold")
    fig.text(
        0.04,
        0.885,
        f"{count} substantial observed regions  ·  {necks} narrow connections  ·  "
        "known camera poses / no reference floor plan",
        color=MUTED,
        fontsize=12,
    )
    for axis in axes:
        axis.set_facecolor(PANEL)
        axis.set(xlim=extent[:2], ylim=extent[2:], aspect="equal", xlabel="World X (m)")
        axis.tick_params(colors=MUTED, labelsize=9)
        axis.xaxis.label.set_color(MUTED)
        axis.yaxis.label.set_color(MUTED)
        for spine in axis.spines.values():
            spine.set_color("#344459")
    axes[0].set_ylabel("World Y (m)")
    axes[0].set_title("Observed occupancy + region partition", color=TEXT, fontsize=13, pad=13)
    axes[0].imshow(image[low[0] : high[0], low[1] : high[1]], origin="lower", extent=extent)
    axes[1].set_title(
        "Region connectivity at measured XY positions", color=TEXT, fontsize=13, pad=13
    )
    nodes = {region["label"]: np.asarray(region["centroid_xy_m"]) for region in summary["regions"]}
    for connection in summary["connections"]:
        a, b = (nodes[label] for label in connection["regions"])
        center = np.asarray(connection["center_xy_m"])
        line = np.vstack((a, center, b))
        axes[1].plot(*line.T, color=MUTED, linewidth=2.4, zorder=1)
        for axis in axes:
            axis.scatter(
                *center, marker="D", s=58, facecolor=BG, edgecolor=TEXT, linewidth=1.2, zorder=3
            )
        axes[1].annotate(
            f"{connection['clearance_width_m']:.2f} m",
            center,
            xytext=(7, -20),
            textcoords="offset points",
            color=MUTED,
            fontsize=9,
        )
    for region in summary["regions"]:
        center = nodes[region["label"]]
        color = palette[region["label"]]
        axes[0].annotate(
            region["label"],
            center,
            ha="center",
            va="center",
            color=TEXT,
            fontsize=12,
            weight="bold",
            bbox={"boxstyle": "round,pad=.3", "fc": BG, "ec": color, "lw": 1.2},
        )
        axes[1].scatter(*center, s=1050, color=color, edgecolor=TEXT, linewidth=1.1, zorder=4)
        axes[1].annotate(
            region["label"],
            center,
            ha="center",
            va="center",
            color=BG,
            fontsize=13,
            weight="bold",
            zorder=5,
        )
        axes[1].annotate(
            f"{region['observed_free_area_m2']:.2f} m² observed",
            center,
            xytext=(0, -34 if center[1] < np.median([xy[1] for xy in nodes.values()]) else 28),
            textcoords="offset points",
            ha="center",
            color=TEXT,
            fontsize=9,
        )
    axes[0].legend(
        handles=[
            Patch(color=OCCUPIED, label="occupied"),
            Patch(color=PANEL, label="unknown"),
            Patch(color=FRAGMENT, label="small fragments"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        frameon=False,
        labelcolor=MUTED,
        fontsize=9,
    )
    axes[1].text(
        0.5,
        -0.19,
        "Lines indicate connectivity; they are not navigation paths.",
        transform=axes[1].transAxes,
        ha="center",
        color=MUTED,
        fontsize=9,
    )
    fig.text(
        0.04,
        0.065,
        f"{summary['small_fragment_count']} small fragments "
        f"({summary['small_fragment_area_m2']:.2f} m² total) retained in the data; "
        "only substantial regions receive graph nodes.",
        color=MUTED,
        fontsize=10,
    )
    fig.text(
        0.04,
        0.028,
        "Regions and free-space necks are geometric candidates, "
        "not semantic room labels or verified doors. "
        "Unknown areas remain unresolved.",
        color=MUTED,
        fontsize=9,
    )
    fig.savefig(output, facecolor=BG)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("vis/multiroom/run_v4"))
    parser.add_argument("--output", type=Path, default=Path("vis/multiroom/report"))
    parser.add_argument("--minimum-area-m2", type=float, default=1.0)
    args = parser.parse_args()
    if not np.isfinite(args.minimum_area_m2) or args.minimum_area_m2 <= 0:
        raise ValueError("Display area threshold must be finite and positive")
    if json.loads((args.run / "status.json").read_text())["status"] != "complete":
        raise ValueError("Visualize a completed frozen run")
    source = args.run / "topology.json"
    topology = json.loads(source.read_text())
    if topology["source"] != "observed_free_space_only":
        raise ValueError("This plot requires topology derived only from observed free space")
    snapshots = sorted((args.run / "maps").glob("station_*.npz"))
    if not snapshots:
        raise ValueError("No acquired map snapshots found")
    with np.load(snapshots[-1]) as archive:
        if not np.allclose(archive["origin"], topology["origin_xy_m"]) or not np.isclose(
            archive["resolution_m"], topology["resolution_m"]
        ):
            raise ValueError("Topology and occupancy calibration disagree")
        states = archive["states"].copy()
    summary = compact_topology(topology, args.minimum_area_m2)
    summary["source_topology_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    summary["source_snapshot_sha256"] = hashlib.sha256(snapshots[-1].read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    draw_topology(topology, states, summary, args.output / "topology.png")
    (args.output / "topology.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "substantial_region_count",
                    "neck_connection_count",
                    "small_fragment_count",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
