"""Time sequential, matched training runs, including imports and cold compiler startup."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from roomgraph.learning.training import atomic_write_json

VARIANTS = {
    "eager": {"compile_model": False, "compile_loss": False},
    "loss": {"compile_model": False, "compile_loss": True},
    "compiled": {"compile_model": True, "compile_loss": True},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/training/headcam_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/runs/efficiency_v1"))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=["eager", "compiled"])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or len(args.variants) != len(set(args.variants)):
        parser.error("Use positive epochs and unique variants")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Benchmark output must be empty; use a new directory for each repetition")
    args.output.mkdir(parents=True, exist_ok=True)
    base = json.loads(args.config.read_text())
    for variant in args.variants:
        config = {
            **base,
            **VARIANTS[variant],
            "epochs": args.epochs,
            "patience": args.epochs,  # Match update counts, without early-stopping confounds.
            "seed": args.seed,
            "run_dir": str(args.output / variant),
        }
        config_path = args.output / f"{variant}.json"
        atomic_write_json(config, config_path)
        environment = os.environ.copy()
        # An isolated, initially empty cache makes first-compilation cost explicit.
        cache = args.output / f"compiler_cache_{variant}"
        environment["TORCHINDUCTOR_CACHE_DIR"] = str(cache.resolve())
        environment["TRITON_CACHE_DIR"] = str((cache / "triton").resolve())
        command = [sys.executable, "scripts/train_edges.py", "--config", str(config_path)]
        print("START", variant, flush=True)
        started = time.perf_counter()
        with (args.output / f"{variant}.log").open("w") as log:
            subprocess.run(
                command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True
            )
        duration = time.perf_counter() - started
        path = Path(config["run_dir"]) / "run.json"
        result = json.loads(path.read_text())
        result.update(launcher_wall_seconds=duration, compiler_cache="isolated; initially empty")
        atomic_write_json(result, path)
        print("COMPLETE", variant, f"{duration:.2f} seconds", flush=True)


if __name__ == "__main__":
    main()
