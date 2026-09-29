"""Compose real camera-path renders with a synchronized map and encode web previews."""

import argparse
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from roomgraph.visualization import camera_triangle, floorplan_pixel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")
    encoders = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, check=True
    ).stdout
    if "libx264" in encoders:
        codec = ["-c:v", "libx264", "-crf", "21"]
    elif "libopenh264" in encoders:
        codec = ["-c:v", "libopenh264", "-b:v", "4M"]
    else:
        raise RuntimeError("FFmpeg needs the libx264 or libopenh264 encoder")
    font = ImageFont.truetype("DejaVuSans.ttf", 22)
    small = ImageFont.truetype("DejaVuSans.ttf", 15)
    for item in json.loads((args.capture / "index.json").read_text()):
        name = item["name"]
        source, output = args.capture / name, args.destination / name
        output.mkdir(parents=True, exist_ok=True)
        frames = source / "film"
        frames.mkdir(exist_ok=True)
        manifest = json.loads((source / "manifest.json").read_text())
        map_image = (
            Image.open(source / "top_down.png")
            .convert("RGB")
            .resize((360, 240), Image.Resampling.LANCZOS)
        )
        extent = manifest["top_down"]["horizontal_extent_m"]
        path = [floorplan_pixel(v["position"][:2], 360, 240, extent) for v in manifest["views"]]
        for index, view in enumerate(manifest["views"]):
            frame = Image.new("RGB", (1152, 576), "#101316")
            draw = ImageDraw.Draw(frame)
            draw.text((18, 15), manifest["scene_config"]["title"], font=font, fill="#edf2ef")
            draw.text((800, 18), "CAMERA PATH / TOP-DOWN", font=small, fill="#a9b8b6")
            with Image.open(source / f"{view['stem']}_overlay.png") as image:
                frame.paste(image, (0, 52))
            local = map_image.copy()
            local_draw = ImageDraw.Draw(local)
            local_draw.line(path + path[:1], fill="#d1a553", width=2)
            _, _, vertices = camera_triangle(view["position"], view["target"])
            points = [floorplan_pixel(v, 360, 240, extent) for v in vertices]
            local_draw.polygon(points, fill="#ffbc59", outline="#101316", width=2)
            frame.paste(local, (780, 130))
            x, y, z = view["position"]
            draw.text(
                (800, 396), f"Position  {x:+.2f}, {y:+.2f}, {z:.2f} m", font=small, fill="#edf2ef"
            )
            draw.text(
                (800, 428),
                f"{index + 1:02d} / {len(manifest['views'])} rendered poses",
                font=small,
                fill="#a9b8b6",
            )
            draw.text((800, 474), "— Visible structure", font=small, fill="#67efc4")
            draw.text((800, 501), "- - Occluded structure", font=small, fill="#ff7a80")
            draw.text((800, 541), "Geometry-derived ground truth", font=small, fill="#a9b8b6")
            frame.save(frames / f"{index:04d}.png")
            if index == 0:
                frame.save(output / "walkthrough-poster.webp", quality=90)
        common = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            str(args.fps),
            "-i",
            str(frames / "%04d.png"),
        ]
        subprocess.run(
            common
            + codec
            + [
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output / "walkthrough.mp4"),
            ],
            check=True,
        )
        subprocess.run(
            common
            + [
                "-filter_complex",
                "[0:v]scale=768:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=3",
                "-loop",
                "0",
                str(output / "walkthrough.gif"),
            ],
            check=True,
        )
        metadata = {
            "scene": name,
            "fps": args.fps,
            "frames": len(manifest["views"]),
            "source_config_sha256": manifest["source_config_sha256"],
            "render": manifest["scene_config"]["render"],
            "cameras": manifest["scene_config"]["cameras"],
        }
        (output / "walkthrough.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Encoded {name}: {len(manifest['views'])} frames at {args.fps} fps", flush=True)


if __name__ == "__main__":
    main()
