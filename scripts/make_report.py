"""Build a self-contained HTML inspection report for a RoomGraph capture."""

import argparse
import base64
import json
from pathlib import Path

import numpy as np
from PIL import Image


def image_uri(path: Path) -> str:
    """Embed an image so the report can be moved or opened without a web server."""
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.capture / "manifest.json").read_text())
    rows = []
    for view in manifest["views"]:
        stem = view["id"]
        with np.load(args.capture / f"{stem}_geometry.npz") as data:
            depth = data["depth"]
        finite = np.isfinite(depth) & (depth > 0)
        normalized = np.where(finite, np.clip(depth / 8, 0, 1), 0)
        color = np.stack([normalized * 60, normalized * 190, normalized * 220], axis=-1)
        path = args.capture / f"{stem}_depth.png"
        Image.fromarray(color.astype(np.uint8)).save(path)
        cards = []
        for suffix, label in [
            ("rgb", "Rendered RGB"),
            ("overlay", "Structural edges"),
            ("depth", "Metric depth · 0–8 m"),
        ]:
            uri = image_uri(args.capture / f"{stem}_{suffix}.png")
            cards.append(f'<figure><img src="{uri}"><figcaption>{label}</figcaption></figure>')
        error = view["depth_p95_absolute_error_m"] * 1000
        rows.append(
            f"<section><h2>{stem} <small>depth error p95: {error:.4f} mm</small></h2>"
            f'<div class="grid">{"".join(cards)}</div></section>'
        )
    template = Path(__file__).with_name("report_template.html").read_text()
    html = template.replace("__SCENE_JSON__", json.dumps(manifest)).replace(
        "__VIEW_ROWS__", "\n".join(rows)
    )
    output = args.capture / "report.html"
    output.write_text(html)
    print(output)


if __name__ == "__main__":
    main()
