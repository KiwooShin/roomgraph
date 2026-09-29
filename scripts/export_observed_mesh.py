"""Export observed planar patches without using reference building geometry."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from roomgraph.learning.training import atomic_write_json
from roomgraph.observed_mesh import observed_wall_patches, write_wall_obj


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    experiment = json.loads(args.results.read_text())
    if experiment.get("status") != "complete":
        raise ValueError("Observed patch export requires a completed acquisition run")
    snapshot = Path(experiment["stations"][-1]["snapshot"])
    started = time.perf_counter()
    with np.load(snapshot) as archive:
        mesh = observed_wall_patches(archive["surface_points"], archive["surface_support"])
    target = args.results.parent / "observed_walls.obj"
    write_wall_obj(target, mesh)
    summary = {
        "status": "complete",
        "input_experiment_sha256": hashlib.sha256(args.results.read_bytes()).hexdigest(),
        "input_snapshot_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        "obj_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "vertices": len(mesh["vertices"]),
        "faces": len(mesh["faces"]),
        "planes": mesh["planes"],
        "patch_size_m": mesh["patch_size_m"],
        "limitations": mesh["limitations"],
        "wall_seconds": time.perf_counter() - started,
    }
    atomic_write_json(summary, args.results.parent / "observed_walls.json")
    print(
        json.dumps({"planes": len(mesh["planes"]), "faces": len(mesh["faces"]), "obj": str(target)})
    )


if __name__ == "__main__":
    main()
