"""Convert completed renderer manifests to a validated model-training cache."""

import argparse
from pathlib import Path

from roomgraph.learning.data import export_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/datasets/headcam_v1"))
    parser.add_argument("--expected-rooms", type=int, default=24)
    args = parser.parse_args()
    manifests = list(args.capture.glob("*/manifest.json"))
    if len(manifests) != args.expected_rooms:
        raise ValueError(f"Expected {args.expected_rooms} complete rooms, found {len(manifests)}")
    print(export_dataset(args.capture, args.output)["splits"])
