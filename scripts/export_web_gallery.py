"""Export a selected, compressed public gallery from a local furnished capture."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from roomgraph.visualization import floorplan_pixel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    scenes = []
    for entry in json.loads((args.capture / "index.json").read_text()):
        name = entry["name"]
        source, output = args.capture / name, args.destination / name
        output.mkdir(exist_ok=True)
        manifest = json.loads((source / "manifest.json").read_text())
        config = manifest["scene_config"]
        width, height = config["render"]["width"], config["render"]["height"]
        extent = manifest["top_down"]["horizontal_extent_m"]
        views = []
        for view, marker in zip(
            manifest["views"], manifest["top_down"]["camera_markers"], strict=True
        ):
            for mode in ("rgb", "overlay"):
                with Image.open(source / f"{view['stem']}_{mode}.png") as image:
                    image.save(output / f"{view['id']}_{mode}.webp", quality=88, method=6)
                    if mode == "overlay":
                        image.resize((420, 280), Image.Resampling.LANCZOS).save(
                            output / f"{view['id']}_thumb.webp", quality=85, method=6
                        )
            points = [floorplan_pixel(v, width, height, extent) for v in marker["vertices_xy"]]
            views.append(
                {
                    "id": view["id"],
                    "position": view["position"],
                    "target": view["target"],
                    "triangle": points,
                }
            )
        with Image.open(source / "top_down.png") as overhead:
            overhead.save(output / "top_down.webp", quality=90, method=6)
            marked = overhead.copy()
            draw = ImageDraw.Draw(marked)
            font = ImageFont.truetype("DejaVuSans.ttf", 25)
            for view in views[:4]:
                draw.polygon(
                    [tuple(p) for p in view["triangle"]], fill="#ffbc59", outline="#131a23", width=3
                )
                origin = floorplan_pixel(view["position"][:2], width, height, extent)
                draw.text(
                    (origin[0], origin[1] + 28),
                    view["id"],
                    font=font,
                    fill="white",
                    stroke_width=3,
                    stroke_fill="#131a23",
                    anchor="mm",
                )
            marked.save(output / "top_down_four.webp", quality=90, method=6)
        (output / "scene.json").write_text(json.dumps(config, indent=2) + "\n")
        scenes.append(
            {
                **entry,
                "description": config["description"],
                "width": width,
                "height": height,
                "views": views,
                "config_sha256": manifest["source_config_sha256"],
            }
        )
    (args.destination / "gallery.json").write_text(json.dumps(scenes, indent=2) + "\n")
    print(f"Exported {len(scenes)} scenes to {args.destination}")


if __name__ == "__main__":
    main()
