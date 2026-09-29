"""Verify deterministic active-view replay while ignoring runtime and output-path differences."""

import argparse
import json
from pathlib import Path


def verify(first, second):
    if any(report.get("status") != "complete" for report in (first, second)):
        raise ValueError("Both experiments must be complete")
    for key in (
        "config",
        "checkpoint_sha256",
        "capture_manifest_sha256",
        "bootstrap_manifest_sha256",
    ):
        if first[key] != second[key]:
            raise ValueError(f"Experiment inputs differ: {key}")
    for report in (first, second):
        if not report.get("observation_cache"):
            raise ValueError("No frozen observation hashes available")
    hashes = [
        sorted(
            (item["source_rgb_sha256"], item["prediction_sha256"])
            for item in report["observation_cache"].values()
        )
        for report in (first, second)
    ]
    if hashes[0] != hashes[1]:
        raise ValueError("Acquired prediction hashes differ")
    if first["initial_reconstruction"]["bounds_m"] != second["initial_reconstruction"]["bounds_m"]:
        raise ValueError("Initial geometry differs")
    policies = [{p["name"]: p for p in report["policies"]} for report in (first, second)]
    if policies[0].keys() != policies[1].keys():
        raise ValueError("Policy sets differ")
    for name, a in policies[0].items():
        b = policies[1][name]
        for key in ("acquired_head_ids", "coverage_auc", "final_metrics"):
            if a[key] != b[key]:
                raise ValueError(f"Policy {name} differs: {key}")
    return {"status": "pass", "observations": len(hashes[0]), "policies": len(policies[0])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(verify(json.loads(args.first.read_text()), json.loads(args.second.read_text())))
    )


if __name__ == "__main__":
    main()
