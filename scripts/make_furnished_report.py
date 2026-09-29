"""Create a local furnished-scene gallery, top-down maps, and overview sheets."""

import argparse
import html
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", nargs="?", type=Path, default=Path("vis/furnished"))
    args = parser.parse_args()
    scenes = json.loads((args.capture / "index.json").read_text())
    sections, navigation = [], []
    overview = Image.new("RGB", (1600, 1160), "#101923")
    plans = Image.new("RGB", (1600, 1160), "#101923")
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 25)
    for index, scene in enumerate(scenes):
        name = scene["name"]
        manifest = json.loads((args.capture / name / "manifest.json").read_text())
        config = manifest["scene_config"]
        title = html.escape(config["title"])
        navigation.append(f'<a href="#{name}">{title}</a>')
        cameras = []
        for view in manifest["views"]:
            stem, camera = view["stem"], view["id"]
            cameras.append(f'''<article class="capture" data-stem="{name}/{stem}">
<div class="capture-head">
<h3>{camera}</h3>
<div class="modes" role="group" aria-label="{camera} display mode">
<button type="button" data-mode="rgb" aria-pressed="true">RGB</button>
<button type="button" data-mode="overlay" aria-pressed="false">Structure + occlusion</button>
</div>
</div>
<a class="full-image" href="{name}/{stem}_rgb.png">
<img src="{name}/{stem}_rgb.png" width="1152"
height="768"
alt="{title}, perspective camera {camera}" loading="lazy">
</a>
<p class="capture-note">Position {tuple(view["position"])} m ·
{view["visible_pixels"]:,} visible / {view["hidden_pixels"]:,} hidden structural-edge pixels</p>
</article>''')
        objects = sorted({p["object_id"].replace("_", " ") for p in manifest["parts"]})
        sections.append(f'''<section class="scene" id="{name}">
<div class="scene-heading">
<div>
<span class="eyebrow">{index + 1:02d} / FURNISHED INTERIORS</span>
<h2>{title}</h2>
<p>{html.escape(config["description"])}</p>
</div>
<span class="count">{manifest["object_count"]} objects · {len(manifest["views"])} cameras</span>
</div>
<div class="scene-overview">
<figure>
<a href="{name}/view_000_rgb.png">
<img src="{name}/view_000_rgb.png"
width="1152"
height="768" alt="{title} furnished interior" loading="lazy">
</a>
<figcaption>Perspective preview · Camera C1</figcaption>
</figure>
<figure>
<a href="{name}/top_down_cameras.png">
<img src="{name}/top_down_cameras.png" width="1152"
height="768"
alt="Top-down {title} with camera position and direction triangles" loading="lazy">
</a>
<figcaption>Top-down · tips indicate viewing direction · ceiling hidden</figcaption>
</figure>
</div>
<div class="scene-links">
<a href="{name}/scene.json">Editable scene JSON</a>
<a href="{name}/manifest.json">Rendered scene manifest</a>
<a href="{name}/room.usda">USD scene</a>
<a href="{name}/top_down.png">Top-down without markers</a>
</div>
<details>
<summary>Furniture inventory</summary>
<p>{html.escape(", ".join(objects))}</p>
</details>
{"".join(cameras)}</section>''')
        for sheet, filename in [(overview, "view_000_rgb.png"), (plans, "top_down_cameras.png")]:
            x, y = (index % 2) * 800, (index // 2) * 580
            with Image.open(args.capture / name / filename) as picture:
                picture = picture.resize((780, 520), Image.Resampling.LANCZOS)
                sheet.paste(picture, (x + 10, y + 50))
            ImageDraw.Draw(sheet).text((x + 20, y + 12), config["title"], font=font, fill="#e9f5fa")
    overview.save(args.capture / "overview.jpg", quality=95)
    plans.save(args.capture / "top_down_overview.jpg", quality=95)
    template = Path(__file__).with_name("furnished_report_template.html").read_text()
    (args.capture / "report.html").write_text(
        template.replace("__NAVIGATION__", "".join(navigation)).replace(
            "__SCENES__", "\n".join(sections)
        )
    )
    print(args.capture / "report.html")


if __name__ == "__main__":
    main()
