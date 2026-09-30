"""Download a small, explicit ARKitScenes raw sequence with local provenance."""

import argparse
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw"
ASSETS = (
    "lowres_wide.zip",
    "lowres_depth.zip",
    "confidence.zip",
    "lowres_wide_intrinsics.zip",
    "lowres_wide.traj",
    "highres_depth.zip",
)


def download(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        temporary = destination.with_suffix(destination.suffix + ".part")
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as out:
            while block := response.read(1024 * 1024):
                out.write(block)
        temporary.replace(destination)
    with destination.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "url": url,
        "bytes": destination.stat().st_size,
        "sha256": checksum,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="42445021")
    parser.add_argument("--split", choices=["Training", "Validation"], default="Validation")
    parser.add_argument("--output", type=Path, default=Path("artifacts/real/arkitscenes"))
    parser.add_argument(
        "--assets",
        nargs="+",
        default=list(ASSETS),
        choices=[*ASSETS, "vga_wide.zip", "vga_wide_intrinsics.zip"],
    )
    args = parser.parse_args()
    if not args.video.isdigit():
        parser.error("Video ID must contain digits only")
    root = args.output / args.split / args.video
    previous = root / "download_receipt.json"
    receipt = json.loads(previous.read_text()) if previous.exists() else {}
    for asset in args.assets:
        target = root / asset
        print(f"Downloading/checking {asset}", flush=True)
        record = download(f"{BASE}/{args.split}/{args.video}/{asset}", target)
        if asset in receipt and record != receipt[asset]:
            raise ValueError(f"Previously receipted asset changed: {asset}")
        receipt[asset] = record
        if asset.endswith(".zip"):
            with zipfile.ZipFile(target) as archive:
                for name in archive.namelist():
                    if not (root / name).resolve().is_relative_to(root.resolve()):
                        raise ValueError("Archive path escapes download directory")
                archive.extractall(root)
    for name, url in {
        "metadata.csv": f"{BASE}/metadata.csv",
        "LICENSE": "https://raw.githubusercontent.com/apple-aiml-research/ARKitScenes/main/LICENSE",
    }.items():
        receipt[name] = download(url, root / name)
    previous.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"root": str(root), "assets": len(receipt)}), flush=True)


if __name__ == "__main__":
    main()
