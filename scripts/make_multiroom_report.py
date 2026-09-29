"""Create a furnished exploration replay and inspectable observed 3D map."""

# HTML/JavaScript template lines stay together for readability.
# ruff: noqa: E501

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageOps

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from make_active_head_report import AMBER, BG, MINT, MUTED, PANEL, TEXT, font

WIDTH, HEIGHT = 1440, 1000


def scatter_map(archive, bounds, output):
    """Draw only acquired points; an empty area stays empty."""
    points, colors = archive["surface_points"], archive["surface_colors"] / 255
    # Cut away ceiling for inspection; retain floor, walls, furnishings and observed door gaps.
    keep = points[:, 2] < 2.75
    points, colors = points[keep], colors[keep]
    stride = max(1, int(np.ceil(len(points) / 22000)))
    points, colors = points[::stride], colors[::stride]
    fig = plt.figure(figsize=(8, 5), dpi=110, facecolor=BG)
    ax = fig.add_subplot(projection="3d", facecolor=BG)
    if len(points):
        ax.scatter(*points.T, c=colors, s=1.9, linewidths=0, depthshade=False)
    edges = archive["edge_points"]
    if len(edges):
        ax.scatter(*edges[::2].T, c=MINT, s=2.2, linewidths=0, depthshade=False)
    ax.set(xlim=bounds[:2], ylim=bounds[2:4], zlim=(0, 3))
    ax.set_box_aspect((bounds[1] - bounds[0], bounds[3] - bounds[2], 3))
    ax.view_init(elev=58, azim=-65)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(output, facecolor=BG)
    plt.close(fig)


def grid_image(archive, bounds, position, path):
    states, origin, resolution = (
        archive["states"],
        archive["origin"],
        float(archive["resolution_m"]),
    )
    colors = np.zeros((*states.shape, 3), np.uint8)
    colors[:] = [30, 43, 59]
    colors[states == 0] = [70, 113, 111]
    colors[states == 1] = [239, 193, 135]
    low = np.floor((np.array([bounds[0], bounds[2]]) - origin) / resolution).astype(int)
    high = np.ceil((np.array([bounds[1], bounds[3]]) - origin) / resolution).astype(int)
    crop = colors[low[1] : high[1], low[0] : high[0]]
    image = Image.fromarray(crop[::-1]).resize((680, 350), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)

    def xy(point):
        return (
            (point[0] - bounds[0]) / (bounds[1] - bounds[0]) * 680,
            (bounds[3] - point[1]) / (bounds[3] - bounds[2]) * 350,
        )

    if len(path) > 1:
        draw.line([xy(p) for p in path], fill=MINT, width=3)
    x, y = xy(position)
    draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=AMBER, outline="white", width=1)
    return image


def top_down_image(reference, topdown, record, path, camera_color=AMBER):
    image = topdown.copy()
    metadata = reference["top_down"]
    center = np.asarray(metadata["center_xy"])
    scale = image.width / metadata["horizontal_extent_m"]

    def xy(point):
        delta = np.asarray(point)[:2] - center
        return (image.width / 2 + delta[0] * scale, image.height / 2 - delta[1] * scale)

    draw = ImageDraw.Draw(image)
    if len(path) > 1:
        draw.line([xy(p) for p in path], fill=MINT, width=max(2, image.width // 200))
    pose = np.asarray(record["camera_to_world"])
    eye, forward = pose[:2, 3], pose[:2, 2]
    forward = forward / max(np.linalg.norm(forward), 1e-8)
    right = np.array([forward[1], -forward[0]])
    triangle = [
        xy(eye + forward * 0.43),
        xy(eye - forward * 0.16 + right * 0.17),
        xy(eye - forward * 0.16 - right * 0.17),
    ]
    draw.polygon(triangle, fill=camera_color, outline="white", width=2)
    draw.line([xy(eye), xy(eye + forward * 0.95)], fill=camera_color, width=2)
    return image


def paste_fit(canvas, image, box):
    fitted = ImageOps.contain(image, (box[2] - box[0], box[3] - box[1]))
    canvas.paste(
        fitted,
        (
            box[0] + (box[2] - box[0] - fitted.width) // 2,
            box[1] + (box[3] - box[1] - fitted.height) // 2,
        ),
    )


def overview_gallery(data, reference, topdown, evaluation, output):
    """One evaluator-named row per visited region: top-down and four acquired overlays."""
    if evaluation is None:
        return
    frames = {frame["id"]: frame for frame in data["trace"]}
    selected, seen = [], set()
    for entry in evaluation["visits"]["region_transition_sequence"]:
        if entry["region"] in seen:
            continue
        seen.add(entry["region"])
        station = frames[entry["frame_id"]]["station"]
        records = [frame for frame in data["trace"] if frame["station"] == station]
        indices = np.linspace(0, min(len(records), 8) - 1, min(4, len(records))).astype(int)
        selected.append((entry["region"], [records[index] for index in indices]))
    image = Image.new("RGB", (1800, 290 * len(selected) + 70), BG)
    draw = ImageDraw.Draw(image)
    draw.text((20, 18), "Acquired views across connected spaces", fill=TEXT, font=font(28, True))
    for row, (name, records) in enumerate(selected):
        y = 70 + row * 290
        draw.text((15, y), f"{name.title()} / reference top-down", fill=MINT, font=font(17, True))
        palette = [MINT, AMBER, "#69baff", "#ef798c"]
        top = topdown.copy()
        for record, color in zip(records, palette, strict=False):
            top = top_down_image(reference, top, record, [], camera_color=color)
        paste_fit(image, top, (8, y + 32, 352, y + 278))
        for column, record in enumerate(records, start=1):
            draw.text(
                (column * 360 + 15, y),
                f"Overlay {column} · {record['id']}",
                fill=palette[column - 1],
                font=font(16),
            )
            paste_fit(
                image,
                Image.open(record["overlay_path"]),
                (column * 360 + 8, y + 32, (column + 1) * 360 - 8, y + 278),
            )
    image.save(output / "overview.jpg", quality=93)


def write_page(output, frames, cloud, summary):
    evaluation = summary.get("evaluation")
    metrics = ""
    if evaluation:
        final, visits = evaluation["final"], evaluation["visits"]
        metrics = (
            f"<p><strong>{visits['rooms_entered']}/{visits['reference_room_count']} rooms entered"
            f" · {final['visible_edge_coverage']:.1%} reference edge visibility"
            f" · {final['edge_precision_10cm']:.1%} 3D edge precision"
            f" · {final['edge_completeness_10cm']:.1%} completeness at 10 cm.</strong> "
            "Visibility and accurate reconstruction are separate measurements.</p>"
        )
    heading = "<h2>Inspect the reconstructed observations</h2>"
    gallery = (
        "<h2>Acquired views in each connected space</h2>"
        '<a href="overview.jpg"><img src="overview.jpg" alt="Four region rows, each with '
        'reference top-down camera orientations and four acquired prediction overlays"></a>'
    )
    extras = ""
    if (output / "topology.png").exists():
        extras += (
            "<h2>Connections inferred from observed free space</h2>"
            '<img src="topology.png" alt="Four substantial observed region candidates with three neck connections">'
            "<p>These are geometric region candidates, not supplied room labels. "
            "Small fragments and unknown boundaries remain recorded.</p>"
        )
    if (output / "observed_walls.obj").exists():
        extras += (
            '<p><a href="observed_walls.obj">Download observed planar patches (OBJ)</a> · '
            '<a href="observed_walls.json">Patch evidence and limitations</a>. '
            "Only supported cells are exported; tall furniture can appear as a plane candidate.</p>"
        )
    page = (
        TEMPLATE.replace("__FRAMES__", json.dumps(frames))
        .replace("__CLOUD__", json.dumps(cloud, separators=(",", ":")))
        .replace(heading, metrics + gallery + heading)
        .replace("<h2>Run details</h2>", extras + "<h2>Run details</h2>")
        .replace(
            "__SUMMARY__",
            json.dumps({k: v for k, v in summary.items() if k != "evaluation"}, indent=2),
        )
    )
    (output / "report.html").write_text(page)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, default=Path("vis/multiroom/run_v1/experiment.json")
    )
    parser.add_argument(
        "--reference", type=Path, default=Path("vis/multiroom/capture/reference.json")
    )
    parser.add_argument("--output", type=Path, default=Path("vis/multiroom/report"))
    parser.add_argument(
        "--refresh-page", action="store_true", help="Reuse existing replay and point data"
    )
    args = parser.parse_args()
    if args.refresh_page:
        summary = json.loads((args.output / "summary.json").read_text())
        if summary.get("status") != "complete":
            raise ValueError("A completed report is required")
        cloud = json.loads((args.output / "cloud.json").read_text())
        frames = [
            str(path.relative_to(args.output))
            for path in sorted((args.output / "replay").glob("frame_*.webp"))
        ]
        write_page(args.output, frames, cloud, summary)
        return
    data = json.loads(args.results.read_text())
    if data.get("status") != "complete":
        raise ValueError("A completed experiment is required")
    reference = json.loads(args.reference.read_text())
    evaluation_path = args.results.parent / "evaluation.json"
    evaluation = json.loads(evaluation_path.read_text()) if evaluation_path.exists() else None
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ("observed_walls.obj", "observed_walls.json"):
        source = args.results.parent / name
        if source.exists():
            shutil.copy2(source, args.output / name)
    replay = args.output / "replay"
    replay.mkdir(exist_ok=True)
    topdown = Image.open(args.reference.parent / "top_down.png").convert("RGB")
    overview_gallery(data, reference, topdown, evaluation, args.output)
    final = np.load(data["stations"][-1]["snapshot"])
    points = final["surface_points"]
    bounds = [
        float(points[:, 0].min() - 0.25),
        float(points[:, 0].max() + 0.25),
        float(points[:, 1].min() - 0.25),
        float(points[:, 1].max() + 0.25),
    ]
    # Display extent comes from the final observed map, never controller input.
    cloud_images, maps = {}, {}
    for station in data["stations"]:
        index = station["station"]
        archive = np.load(station["snapshot"])
        target = args.output / f"map_{index:03d}.png"
        scatter_map(archive, bounds, target)
        cloud_images[index] = Image.open(target).convert("RGB")
        maps[index] = archive
    accumulated_path = [data["path"][0]]
    station_paths = {}
    for station in data["stations"]:
        station_paths[station["station"]] = list(accumulated_path)
        accumulated_path.extend(station.get("executed_path", [])[1:])
    last_ids = {s["view_ids"][-1]: s["station"] for s in data["stations"] if s["view_ids"]}
    map_station = None
    frames = []
    for index, record in enumerate(data["trace"]):
        if record["id"] in last_ids:
            map_station = last_ids[record["id"]]
        canvas = Image.new("RGB", (WIDTH, HEIGHT), BG)
        draw = ImageDraw.Draw(canvas)
        draw.text((24, 20), "ROOMGRAPH / EXPLORING CONNECTED ROOMS", fill=MINT, font=font(15, True))
        draw.text(
            (24, 48),
            "Build a shared map, one observation at a time",
            fill=TEXT,
            font=font(32, True),
        )
        station = record["station"]
        draw.text(
            (24, 94),
            f"Station {station + 1}  ·  frame {index + 1}/{len(data['trace'])}  ·  travelled {record['path_length_m']:.1f} m  ·  sensor-assisted pilot",
            fill=MUTED,
            font=font(16),
        )
        panels = [
            (20, 135, 480, 500),
            (490, 135, 950, 500),
            (960, 135, 1420, 500),
            (20, 515, 710, 935),
            (720, 515, 1420, 935),
        ]
        for box in panels:
            draw.rounded_rectangle(box, radius=15, fill=PANEL)
        titles = [
            "Reference building / camera direction",
            "Acquired head-camera RGB",
            "Predicted visible structural edges",
            "Observed navigation map",
            "Accumulated observed 3D geometry",
        ]
        for box, title in zip(panels, titles, strict=True):
            draw.text((box[0] + 12, box[1] + 12), title, fill=TEXT, font=font(16, True))
        top = top_down_image(reference, topdown, record, station_paths[station])
        paste_fit(canvas, top, (30, 185, 470, 480))
        paste_fit(canvas, Image.open(record["rgb_path"]), (500, 185, 940, 480))
        paste_fit(canvas, Image.open(record["overlay_path"]), (970, 185, 1410, 480))
        draw.text((32, 165), "Reference is shown for evaluation only", fill=MUTED, font=font(12))
        if map_station is not None:
            observed = maps[map_station]
            navigation = grid_image(
                observed,
                bounds,
                record["request"]["robot_base"]["position"],
                station_paths[station],
            )
            paste_fit(canvas, navigation, (30, 565, 700, 910))
            paste_fit(canvas, cloud_images[map_station], (730, 555, 1410, 920))
        else:
            draw.text((45, 695), "Acquiring the first local sweep…", fill=MUTED, font=font(20))
            draw.text((745, 695), "No completed map snapshot yet", fill=MUTED, font=font(20))
        draw.text(
            (32, 548),
            "Teal: observed free  ·  amber: obstacles  ·  dark: unknown",
            fill=MUTED,
            font=font(12),
        )
        draw.text(
            (733, 548),
            "RGB-D surfaces + learned edge points; ceiling hidden for inspection",
            fill=MUTED,
            font=font(12),
        )
        draw.text(
            (24, 950),
            "Known poses + ideal depth · no floor-plan access by planner · kinematic movement · map panels update after each local sweep",
            fill=AMBER,
            font=font(14),
        )
        draw.text(
            (24, 976),
            "Replay compresses acquisition time; the head views are actual renders. Missing surfaces remain absent.",
            fill=MUTED,
            font=font(12),
        )
        frame = replay / f"frame_{index:04d}.webp"
        canvas.save(frame, quality=86)
        frames.append(str(frame.relative_to(args.output)))
    Image.open(replay / f"frame_{len(frames) - 1:04d}.webp").save(
        args.output / "poster.jpg", quality=92
    )
    # Compact public inspectable map: deterministic sampling, never reference geometry.
    indices = np.arange(0, len(points), max(1, int(np.ceil(len(points) / 22000))))
    edge_points = final["edge_points"]
    edge_indices = np.arange(0, len(edge_points), max(1, int(np.ceil(len(edge_points) / 10000))))
    cloud = {
        "surface": np.column_stack([points[indices], final["surface_colors"][indices]])
        .round(3)
        .tolist(),
        "edges": edge_points[edge_indices].round(3).tolist(),
        "path": data["path"],
    }
    (args.output / "cloud.json").write_text(json.dumps(cloud, separators=(",", ":")))
    summary = {
        "status": "complete",
        "sensor_assumptions": data["config"]["sensor_assumptions"],
        "stop_reason": data["stop_reason"],
        "frames": len(data["trace"]),
        "stations": len(data["stations"]),
        "path_length_m": data["path_length_m"],
        "action_seconds": data["action_seconds"],
        "wall_seconds": data["wall_seconds"],
        "surface_voxels": len(points),
        "edge_voxels": len(edge_points),
        "experiment_sha256": hashlib.sha256(args.results.read_bytes()).hexdigest(),
        "evaluation": evaluation,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_page(args.output, frames, cloud, summary)
    print(
        json.dumps(
            {
                "report": str(args.output / "report.html"),
                "frames": len(frames),
                "surface_voxels": len(points),
                "edge_voxels": len(edge_points),
            }
        )
    )


TEMPLATE = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RoomGraph — Connected rooms</title>
<style>body{margin:0;background:#0c1420;color:#e7f0f7;font:16px system-ui}main{max-width:1440px;margin:auto;padding:24px}h1{font-size:clamp(28px,5vw,52px)}p{max-width:90ch;line-height:1.6;color:#a9bed0}a{color:#55e4bd}img,canvas{width:100%;border-radius:14px;background:#142131}canvas{height:620px;touch-action:none}button,input{accent-color:#55e4bd}button{padding:10px 20px;background:#55e4bd;border:0;border-radius:8px;margin:12px;color:#09201a}input[type=range]{width:65%}pre{white-space:pre-wrap;overflow-wrap:anywhere;padding:20px;background:#142131}label{margin:12px}@media(max-width:600px){canvas{height:380px}main{padding:12px}}</style><main>
<p>ROOMGRAPH / MULTI-ROOM EXPLORATION</p><h1>From one room to a shared indoor map.</h1><p>A head camera looks around, builds a partial map, and chooses reachable boundaries of unknown space. It starts in one furnished room and accumulates acquired observations in a common coordinate frame. This pilot supplies ideal depth and known poses; the controller receives no reference floor plan.</p>
<img id="replay" alt="Furnished building camera position, acquired RGB and edge prediction, partial occupancy map and accumulated observed geometry"><div><button id="play">Play</button><input id="slider" aria-label="Acquired frame" type="range" min="0" value="0"><span id="counter"></span></div><p>Map panels update after each local sweep. Replay compresses acquisition time; it does not represent a dynamically walking humanoid. The reference building panel is for evaluation only.</p>
<h2>Inspect the reconstructed observations</h2><p>Drag to orbit; scroll to zoom. Surfaces come from acquired RGB-D observations; green structural points require the learned visible and typed edge channels. Neither view fills in unobserved surfaces. Ceiling is hidden for inspection.</p><label><input id="surfaces" type="checkbox" checked> RGB-D surfaces</label><label><input id="edges" type="checkbox" checked> Learned edges</label><canvas id="cloud" aria-label="Interactive accumulated observed 3D point cloud"></canvas>
<p><a href="summary.json">Measured summary</a> · <a href="cloud.json">Selected 3D point data</a></p><h2>Run details</h2><pre>__SUMMARY__</pre></main>
<script>const frames=__FRAMES__,data=__CLOUD__;let index=0,timer=null;const image=document.getElementById('replay'),slider=document.getElementById('slider');slider.max=frames.length-1;function frame(){image.src=frames[index];slider.value=index;document.getElementById('counter').textContent=`${index+1}/${frames.length}`;}slider.oninput=()=>{index=+slider.value;frame()};document.getElementById('play').onclick=()=>{if(timer){clearInterval(timer);timer=null}else{timer=setInterval(()=>{index=(index+1)%frames.length;frame()},150)}document.getElementById('play').textContent=timer?'Pause':'Play'};frame();
const canvas=document.getElementById('cloud'),ctx=canvas.getContext('2d');let yaw=-.6,pitch=.9,zoom=1,drag=null;const pts=data.surface.filter(p=>p[2]<2.75);let center=[0,0,1];for(let k=0;k<2;k++)center[k]=pts.reduce((s,p)=>s+p[k],0)/Math.max(pts.length,1);function draw(){canvas.width=canvas.clientWidth*devicePixelRatio;canvas.height=canvas.clientHeight*devicePixelRatio;const w=canvas.width,h=canvas.height;ctx.fillStyle='#142131';ctx.fillRect(0,0,w,h);function project(p){let x=p[0]-center[0],y=p[1]-center[1],z=p[2]-center[2],a=x*Math.cos(yaw)-y*Math.sin(yaw),b=x*Math.sin(yaw)+y*Math.cos(yaw);return[w/2+a*w*.065*zoom,h/2-(z*Math.cos(pitch)+b*Math.sin(pitch))*w*.065*zoom,b*Math.cos(pitch)-z*Math.sin(pitch)]}const items=[];if(document.getElementById('surfaces').checked)for(const p of pts)items.push([project(p),`rgb(${p[3]|0},${p[4]|0},${p[5]|0})`,2]);if(document.getElementById('edges').checked)for(const p of data.edges)items.push([project(p),'#55e4bd',2.5]);items.sort((a,b)=>a[0][2]-b[0][2]);for(const [p,c,s] of items){ctx.fillStyle=c;ctx.fillRect(p[0],p[1],s*devicePixelRatio,s*devicePixelRatio)}}canvas.onpointerdown=e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId)};canvas.onpointermove=e=>{if(!drag)return;yaw+=(e.clientX-drag[0])*.008;pitch=Math.max(-1.5,Math.min(1.5,pitch+(e.clientY-drag[1])*.008));drag=[e.clientX,e.clientY];draw()};canvas.onpointerup=()=>drag=null;canvas.onwheel=e=>{e.preventDefault();zoom=Math.max(.3,Math.min(3,zoom*Math.exp(-e.deltaY*.001)));draw()};window.onresize=draw;document.getElementById('surfaces').onchange=draw;document.getElementById('edges').onchange=draw;draw();</script></html>"""


if __name__ == "__main__":
    main()
