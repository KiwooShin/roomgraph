"""Prepare deterministic, room-disjoint furnished benchmark configurations."""

import argparse
import json
import math
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmark-configs"))
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for family in ["bedroom", "kitchen", "living", "office"]:
        for variant in range(6):
            config = json.loads(Path(f"configs/scenes/{family}.json").read_text())
            split = "train" if variant < 4 else ("val" if variant == 4 else "test")
            name = f"{family}_{variant:02d}"
            dimensions = np.round(
                [rng.uniform(6.0, 7.6), rng.uniform(5.0, 6.4), rng.uniform(2.8, 3.5)], 3
            )
            config.update(
                name=name,
                title=f"{family.title()} variant {variant}",
                seed=int(rng.integers(1, 1000000)),
            )
            config["dataset"] = {
                "building_id": name,
                "split": split,
                "family": family,
                "benchmark": "rooms_v1",
                "seed": args.seed,
            }
            config["room"].update(
                template="scaled_room_v1",
                dimensions_m=dimensions.tolist(),
                accent_color=rng.uniform(0.18, 0.65, 3).round(3).tolist(),
            )
            config["lighting"].update(
                daylight_intensity=int(rng.integers(550, 1200)),
                ceiling_intensity=int(rng.integers(1000, 2000)),
                color_temperature_kelvin=int(rng.integers(4200, 6500)),
            )
            config["render"].update(width=576, height=384, samples_per_pixel=16)
            config["headcam"] = {
                "enabled": True,
                "model": "original_neo_style_proxy_v1",
                "arm_reach_m": float(rng.uniform(0.43, 0.62)),
                "camera_height_m": 1.585,
                "note": "Original proxy, not official 1X asset or camera calibration",
            }
            config["cameras"] = []
            for i in range(24):
                angle = 2 * math.pi * (i + rng.uniform(-0.3, 0.3)) / 24
                eye = [
                    float(dimensions[0] / 2 * 0.68 * math.cos(angle)),
                    float(dimensions[1] / 2 * 0.64 * math.sin(angle)),
                    1.585,
                ]
                target = [
                    float(rng.uniform(-0.5, 0.5)),
                    float(rng.uniform(-0.5, 0.5)),
                    float(rng.uniform(0.80, 1.25)),
                ]
                config["cameras"].append(
                    {
                        "id": f"C{i:02d}",
                        "position": eye,
                        "target": target,
                        "focal_length_mm": float(rng.uniform(12, 17)),
                    }
                )
            (args.output / f"{name}.json").write_text(json.dumps(config, indent=2) + "\n")
    print("24 rooms: 16 train / 4 val / 4 test; 24 views each")


if __name__ == "__main__":
    main()
