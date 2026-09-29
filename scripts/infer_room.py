"""Infer a rectangular room from RGB captures and known camera calibration only."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from roomgraph.learning.model import EdgeNet
from roomgraph.learning.reconstruction import fit_room, write_room_obj


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest",
        type=Path,
        help="Capture manifest with views, image stems, intrinsics and camera_to_world",
    )
    parser.add_argument("--run", type=Path, default=Path("artifacts/runs/headcam_v1"))
    parser.add_argument("--output", type=Path, default=Path("vis/inferred_room"))
    parser.add_argument("--views", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    settings = json.loads((args.run / "inference.json").read_text())
    manifest = json.loads(args.manifest.read_text())
    checkpoint = torch.load(args.run / "best.pt", map_location="cpu", weights_only=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    model = EdgeNet(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    records = []
    probabilities = []
    selected = np.linspace(
        0, len(manifest["views"]) - 1, min(args.views, len(manifest["views"])), dtype=int
    )
    with torch.inference_mode():
        for index in selected:
            view = manifest["views"][index]
            image = cv2.cvtColor(
                cv2.imread(str(args.manifest.parent / f"{view['stem']}_rgb.png")), cv2.COLOR_BGR2RGB
            )
            height, width = image.shape[:2]
            intrinsic = np.asarray(view["intrinsics"]).copy()
            intrinsic[0] *= 384 / width
            intrinsic[1] *= 256 / height
            tensor = (
                torch.from_numpy(
                    cv2.resize(image, (384, 256), interpolation=cv2.INTER_AREA)
                    .transpose(2, 0, 1)
                    .copy()
                )
                .unsqueeze(0)
                .to(device)
                .float()
                / 255
            )
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                probability = model(tensor).float().sigmoid()[0].cpu().numpy()
            probabilities.append(probability)
            records.append(
                {"intrinsics": intrinsic.tolist(), "camera_to_world": view["camera_to_world"]}
            )
    result = fit_room(np.stack(probabilities), records, settings["threshold"])
    (args.output / "room.json").write_text(json.dumps(result, indent=2) + "\n")
    write_room_obj(result["bounds_m"], args.output / "room.obj")
    np.savez_compressed(
        args.output / "edge_probabilities.npz",
        probabilities=np.asarray(probabilities, dtype=np.float16),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
