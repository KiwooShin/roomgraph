"""Create camera-path configurations for a continuous furnished-room walkthrough."""

import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=96)
    parser.add_argument("--output", type=Path, default=Path("artifacts/motion-configs"))
    args = parser.parse_args()
    if args.frames < 8:
        parser.error("--frames must be at least 8")
    args.output.mkdir(parents=True, exist_ok=True)
    for path in sorted(Path("configs/scenes").glob("*.json")):
        config = json.loads(path.read_text())
        config["render"].update(width=768, height=512, samples_per_pixel=32)
        config["cameras"] = []
        for index in range(args.frames):
            angle = -math.pi / 2 - math.pi / 4 + 2 * math.pi * index / args.frames
            config["cameras"].append(
                {
                    "id": f"F{index:03d}",
                    "position": [
                        round(2.05 * math.cos(angle), 6),
                        round(1.45 * math.sin(angle), 6),
                        1.95,
                    ],
                    "target": [0, 0, 1.05],
                    "focal_length_mm": 13,
                }
            )
        (args.output / path.name).write_text(json.dumps(config, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
