"""Local visual report of frozen real-data predictions and calibration sensitivity."""

# HTML/JavaScript templates and human-readable labels intentionally use long lines.
# ruff: noqa: E501

import argparse
import html
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from check_arkitscenes_replay import verify_manifest
from PIL import Image, ImageDraw, ImageFont
from run_arkitscenes import digest

BG = "#0c1420"
INK = "#e7f0f7"
MINT = "#55e4bd"
AMBER = "#efc187"


def load(path):
    return json.loads(path.read_text())


def grouped(summary):
    groups = {}
    for row in summary["results"]:
        groups.setdefault((row["variant"], row["fusion"]), []).append(row)
    return groups


def chart(summary, target):
    groups = grouped(summary)
    names = list(dict.fromkeys(r["variant"] for r in summary["results"]))
    with plt.rc_context(
        {
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.edgecolor": INK,
        }
    ):
        fig, ax = plt.subplots(figsize=(13, 6.5))
        for offset, method, color in (
            (-0.18, "all_valid", AMBER),
            (0.18, "confidence_multiview", MINT),
        ):
            values = [
                [r["surface_metrics"]["f1_10cm"] * 100 for r in groups[(name, method)]]
                for name in names
            ]
            ax.barh(
                np.arange(len(names)) + offset,
                [np.mean(v) for v in values],
                height=0.32,
                xerr=[np.std(v) for v in values],
                color=color,
                label=method.replace("_", " "),
            )
        ax.set_yticks(np.arange(len(names)), [n.replace("_", " ") for n in names])
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("Surface F1 @ 10 cm against available reference-depth views (%)")
        ax.set_title(
            "Sensitivity to injected camera and depth errors", loc="left", fontsize=18, pad=15
        )
        ax.legend(facecolor=BG, labelcolor=INK, loc="lower left")
        ax.grid(axis="x", alpha=0.15)
        fig.tight_layout()
        fig.savefig(target, dpi=140)
        plt.close(fig)


def camera_map(surface, records, width=420, height=420):
    points = surface["points"]
    centers = np.array([r["camera_to_world"] for r in records])[:, :3, 3]
    xy = np.concatenate([points[:, :2], centers[:, :2]])
    low, high = np.quantile(xy, 0.005, axis=0), np.quantile(xy, 0.995, axis=0)
    scale = min(
        (width - 40) / max(0.1, high[0] - low[0]), (height - 40) / max(0.1, high[1] - low[1])
    )
    middle = (low + high) / 2

    def project(values):
        return (values - middle) * [scale, -scale] + [width / 2, height / 2]

    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)
    selected = np.arange(0, len(points), max(1, len(points) // 15000))
    for pixel, color in zip(
        project(points[selected, :2]), surface["colors"][selected], strict=True
    ):
        x, y = pixel
        draw.rectangle((x, y, x + 1, y + 1), fill=tuple(np.clip(color * 0.55, 0, 255).astype(int)))
    path = project(centers[:, :2])
    draw.line([tuple(p) for p in path], fill="#657689", width=2)
    return canvas, project, path


def marker(image, project, record, color=MINT):
    pose = np.array(record["camera_to_world"])
    direction = pose[:2, 2]
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return
    direction /= norm
    center = pose[:2, 3]
    side = np.array([-direction[1], direction[0]])
    triangle = np.array(
        [
            center + direction * 0.20,
            center - direction * 0.10 + side * 0.10,
            center - direction * 0.10 - side * 0.10,
        ]
    )
    ImageDraw.Draw(image).polygon([tuple(x) for x in project(triangle)], fill=color, outline=INK)


def media(root, output, manifest, surface):
    records = manifest["frames"]
    base, project, path = camera_map(surface, records)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    sheet = Image.new("RGB", (1800, 4 * 290 + 60), BG)
    ImageDraw.Draw(sheet).text(
        (20, 15),
        "Real ARKitScenes capture · recorded camera path + frozen-model overlays",
        font=font,
        fill=INK,
    )
    for row, start in enumerate(np.linspace(0, len(records) - 8, 4).astype(int)):
        mapping = base.copy()
        for col, i in enumerate(range(start, start + 8, 2)):
            marker(mapping, project, records[i], [MINT, AMBER, "#8fbcff", "#ee90bb"][col])
            overlay = Image.open(root / "predictions" / f"{records[i]['id']}_prediction.png")
            sheet.paste(overlay.resize((350, 234)), (365 + col * 358, 60 + row * 290 + 32))
            elapsed = records[i]["timestamp_s"] - records[0]["timestamp_s"]
            ImageDraw.Draw(sheet).text(
                (365 + col * 358, 60 + row * 290),
                f"F{i:03d} · {elapsed:.2f} s",
                font=small,
                fill=INK,
            )
        sheet.paste(mapping.resize((350, 250)), (8, 60 + row * 290 + 16))
    sheet.save(output / "overview.jpg", quality=92)
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        "1200x520",
        "-r",
        "5",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libopenh264",
        "-b:v",
        "1600k",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output / "replay.mp4"),
    ]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
        for i, record in enumerate(records):
            frame = Image.new("RGB", (1200, 520), BG)
            draw = ImageDraw.Draw(frame)
            draw.text(
                (18, 12),
                "ARKitScenes / real capture / frozen synthetic edge model",
                font=font,
                fill=INK,
            )
            mapping = base.copy()
            if i:
                ImageDraw.Draw(mapping).line([tuple(p) for p in path[: i + 1]], fill=AMBER, width=3)
            marker(mapping, project, record)
            frame.paste(mapping, (12, 55))
            for x, suffix, title in (
                (434, "rgb", "Upright RGB"),
                (818, "prediction", "Predicted visible structural edges"),
            ):
                frame.paste(
                    Image.open(root / "predictions" / f"{record['id']}_{suffix}.png").resize(
                        (372, 248)
                    ),
                    (x, 130),
                )
                draw.text((x, 94), title, font=small, fill=INK)
            elapsed = record["timestamp_s"] - records[0]["timestamp_s"]
            draw.text(
                (20, 490),
                f"F{i:03d} / {len(records)}   capture t={elapsed:.2f}s   |   Map: final observed surfaces; triangle: recorded camera",
                font=small,
                fill=INK,
            )
            if i == 0:
                frame.save(output / "poster.jpg", quality=90)
            process.stdin.write(frame.tobytes())
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("Video encoding failed")


VIEWER = """
const canvas=document.querySelector('#cloud'),ctx=canvas.getContext('2d'),select=document.querySelector('#variant');
let yaw=-.6,pitch=.8,zoom=1,drag=null;
function draw(){const d=clouds[select.value];const w=canvas.width=canvas.clientWidth*devicePixelRatio,h=canvas.height=450*devicePixelRatio;
ctx.fillStyle='#142131';ctx.fillRect(0,0,w,h);const center=d.center,s=Math.min(w,h*1.5)*.7/d.span*zoom;
function p(a){let x=a[0]-center[0],y=a[1]-center[1],z=a[2]-center[2],u=x*Math.cos(yaw)-y*Math.sin(yaw),v=x*Math.sin(yaw)+y*Math.cos(yaw);return [w/2+u*s,h/2-(v*Math.sin(pitch)+z*Math.cos(pitch))*s,v*Math.cos(pitch)-z*Math.sin(pitch)]}
const all=[];if(document.querySelector('#surfaces').checked)for(const a of d.surface)all.push([p(a),`rgb(${a[3]},${a[4]},${a[5]})`,2]);
if(document.querySelector('#edges').checked)for(const a of d.edges)all.push([p(a),'#55e4bd',3]);all.sort((a,b)=>a[0][2]-b[0][2]);for(const [q,c,r]of all){ctx.fillStyle=c;ctx.fillRect(q[0],q[1],r*devicePixelRatio,r*devicePixelRatio)}
ctx.strokeStyle='#efc187';ctx.lineWidth=2*devicePixelRatio;ctx.beginPath();d.path.forEach((a,i)=>{let q=p(a);i?ctx.lineTo(q[0],q[1]):ctx.moveTo(q[0],q[1])});ctx.stroke();}
canvas.onpointerdown=e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId)};
canvas.onpointermove=e=>{if(!drag)return;yaw+=(e.clientX-drag[0])*.008;pitch=Math.max(-1.5,Math.min(1.57,pitch+(e.clientY-drag[1])*.008));drag=[e.clientX,e.clientY];draw()};canvas.onpointerup=canvas.onpointercancel=()=>drag=null;
canvas.onwheel=e=>{e.preventDefault();zoom=Math.max(.3,Math.min(4,zoom*Math.exp(-e.deltaY*.001)));draw()};
canvas.onkeydown=e=>{if(e.key==='ArrowLeft')yaw-=.1;else if(e.key==='ArrowRight')yaw+=.1;else if(e.key==='ArrowUp')pitch=Math.min(1.57,pitch+.1);else if(e.key==='ArrowDown')pitch=Math.max(-1.5,pitch-.1);else return;e.preventDefault();draw()};
document.querySelector('#top').onclick=()=>{yaw=0;pitch=Math.PI/2;draw()};document.querySelector('#reset').onclick=()=>{yaw=-.6;pitch=.8;zoom=1;draw()};
for(const e of [select,document.querySelector('#surfaces'),document.querySelector('#edges')])e.onchange=draw;new ResizeObserver(draw).observe(canvas);draw();
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("vis/arkitscenes/pilot_v2"))
    args = parser.parse_args()
    root, output = args.run, args.run / "report"
    output.mkdir(exist_ok=True)
    summary, manifest = load(root / "summary.json"), verify_manifest(root)
    if summary["status"] != "complete":
        raise ValueError("A completed experiment is required")
    replay = load(root / "replay_check.json")
    if replay["status"] != "pass" or replay["summary_sha256"] != digest(root / "summary.json"):
        raise ValueError("A matching successful cached replay is required")
    chart(summary, output / "sensitivity.png")
    nominal = np.load(root / "nominal_11_confidence_multiview.npz")
    media(root, output, manifest, nominal)
    clouds = {}
    for row in summary["results"]:
        if row["variant"] not in ("nominal", "jitter_5cm_3deg") or row["seed"] != 11:
            continue
        with np.load(root / f"{row['id']}.npz") as archive:
            points, colors = archive["points"], archive["colors"]
            ids = np.linspace(0, len(points) - 1, min(18000, len(points))).astype(int)
            clouds[row["id"]] = {
                "surface": np.column_stack(
                    [np.round(points[ids], 3), colors[ids].astype(int)]
                ).tolist(),
                "edges": np.round(archive["edge_points"], 3).tolist(),
                "path": [
                    np.array(r["camera_to_world"])[:3, 3].tolist() for r in manifest["frames"]
                ],
                "center": np.median(points, axis=0).tolist(),
                "span": float(max(3, np.ptp(points, axis=0).max())),
            }
    rows = []
    for (variant, method), values in grouped(summary).items():
        scores = [
            np.mean([r["surface_metrics"][key] for r in values]) * 100
            for key in ("precision_10cm", "recall_10cm", "f1_10cm")
        ]
        rows.append(
            f"<tr><td>{variant}</td><td>{method}</td><td>{len(values)}</td>"
            + "".join(f"<td>{v:.2f}%</td>" for v in scores)
            + "</tr>"
        )
    options = "".join(f'<option value="{k}">{k.replace("_", " ")}</option>' for k in clouds)
    limits = "".join(f"<li>{html.escape(x)}</li>" for x in summary["limitations"])
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RoomGraph / Real capture and calibration sensitivity</title>
<style>body{{margin:0;background:{BG};color:{INK};font:17px/1.65 system-ui}}main{{max-width:1200px;margin:auto;padding:30px 20px}}h1{{font-size:clamp(30px,5vw,54px);line-height:1.1}}h2{{margin-top:50px}}a{{color:{MINT}}}.eyebrow{{color:{MINT};letter-spacing:.15em}}img,video,canvas{{width:100%;border-radius:10px}}canvas{{height:450px;touch-action:none;background:#142131}}.callout{{border-left:3px solid {AMBER};padding:15px 25px;background:#142131}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:9px;border-bottom:1px solid #304050;text-align:left;white-space:nowrap}}button,select{{background:#22374a;color:{INK};padding:10px;border:1px solid #567;border-radius:5px;max-width:100%}}.controls{{display:flex;flex-wrap:wrap;gap:12px;margin:15px 0}}small{{color:#afc1cf}}</style>
<main><p class="eyebrow">ROOMGRAPH / REAL-WORLD PILOT</p><h1>From simulated rooms to a real RGB-D capture.</h1>
<p>ARKitScenes validation capture 42445021 · {summary["frames"]} input views · frozen synthetic edge model · measured depth and supplied estimated camera poses.</p>
<div class="callout"><strong>Calibration is part of the experiment.</strong> We compare nominal estimates with deliberately injected pose, calibration, depth-scale and synchronization errors. Confidence + multi-view filtering rejects unsupported observations; it does not correct drift.</div>
<h2>Recorded movement and learned overlays</h2><video controls playsinline preload="metadata" poster="poster.jpg" src="replay.mp4"></video><p><small>Five selected frames per playback second; original timestamps are displayed. The top-down background is the final observed map, not a claim of progressive complete reconstruction. Images are rotated upright using dataset metadata, with matching camera calibration.</small></p>
<img src="overview.jpg" loading="lazy" alt="Four rows of camera-position triangles and four learned edge overlays per row">
<h2>Inspect the observed 3D map</h2><p>Drag or use arrow keys to rotate; scroll to zoom. Mint points are learned structural-edge predictions. Colored surfaces come from measured depth.</p><div class="controls"><select id="variant" aria-label="Reconstruction variant">{options}</select><label><input type="checkbox" id="surfaces" checked>Surfaces</label><label><input type="checkbox" id="edges" checked>Edges</label><button id="top">Top-down</button><button id="reset">Reset</button></div><canvas id="cloud" tabindex="0" aria-label="Interactive observed reconstruction"></canvas>
<h2>What happens when calibration is wrong?</h2><img src="sensitivity.png" alt="Reconstruction sensitivity to injected errors"><p>Jitter uses three fixed seeds; bars show means and standard deviations. Other cases are deterministic. Translation and rotation jitter scales are per axis. The same RGB predictions and selected input frames are reused in every replay.</p>
<div class="callout">These are <strong>surface-agreement scores against the available reference-depth subset</strong>, not architectural-edge accuracy or full-building reconstruction scores. Reference views are withheld from fusion; their world placement still uses supplied ARKit pose estimates. Correct reconstructed surfaces outside reference coverage can lower precision.</div>
<div class="scroll"><table><thead><tr><th>Error</th><th>Fusion</th><th>Seeds</th><th>Precision @10cm</th><th>Recall @10cm</th><th>F1 @10cm</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>
<p>Measured complete experiment: {summary["wall_seconds"]:.1f}s, including input validation, inference, reference construction and {len(summary["results"])} fusion evaluations; media generation excluded. Median model forward: {summary["median_inference_seconds"] * 1000:.2f}ms. No retraining.</p>
<h2>Scope and next steps</h2><ul>{limits}</ul><p>The next correction stage should jointly refine overlapping-frame poses using robust geometric residuals, retain pose priors in ambiguous views, and re-fuse original observations after corrections. Persistent calibration bias needs calibration refinement; averaging alone cannot remove it.</p>
<p>Data: <a href="https://github.com/apple-aiml-research/ARKitScenes">ARKitScenes, Baruch et al. (2021)</a>. <a href="https://github.com/apple-aiml-research/ARKitScenes/blob/main/LICENSE">Dataset license</a>. <a href="https://github.com/KiwooShin/roomgraph">RoomGraph source</a>. Full captures, predictions and visualization artifacts remain local.</p></main>
<script>const clouds={json.dumps(clouds, separators=(",", ":"))};{VIEWER}</script></html>"""
    (output / "report.html").write_text(page)
    print(output / "report.html")


if __name__ == "__main__":
    main()
