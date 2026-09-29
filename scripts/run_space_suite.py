"""Run frozen multi-space RGB-D pilots sequentially, with isolated renderer cleanup.

Building descriptions enter only the rendering/evaluation commands. The existing
controller gets the unchanged sensor protocol and a whitelisted experiment config.
Completed cases may be resumed only when input and result hashes still match;
failed/partial cases remain intact and require a fresh output for another attempt.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_KEYS = {
    "name",
    "checkpoint",
    "network_size",
    "threshold",
    "robot_arm_reach_m",
    "start_position",
    "initial_body_yaw_deg",
    "map_bounds_xy",
    "map_resolution_m",
    "footprint_radius_m",
    "max_stations",
    "max_frames",
    "step_distance_m",
    "body_speed_m_s",
    "body_yaw_speed_deg_s",
    "head_pitch_deg",
    "head_yaw_speed_deg_s",
    "head_pitch_speed_deg_s",
    "focal_length_mm",
    "sensor_assumptions",
}
CASE_KEYS = {
    "id",
    "title",
    "building",
    "split",
    "start_position",
    "initial_body_yaw_deg",
    "description",
}
PIPELINE_FILES = (
    "installation/run_python.sh",
    "scripts/run_space_suite.py",
    "scripts/run_multiroom.py",
    "scripts/render_building.py",
    "scripts/run_active_head.py",
    "scripts/evaluate_multiroom.py",
    "scripts/check_multiroom_replay.py",
    "scripts/export_observed_mesh.py",
    "scripts/make_multiroom_topology.py",
    "scripts/make_multiroom_report.py",
)


class FrozenInputError(ValueError):
    """A frozen input changed; no further cases may run under this provenance."""


class RendererCleanupError(RuntimeError):
    """GPU container termination is uncertain; do not launch another renderer."""


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def local_path(path, root=ROOT):
    """Docker paths must be project relative; prohibit outside/symlink escapes."""
    path = Path(path)
    target = (root / path).resolve() if not path.is_absolute() else path.resolve()
    return target.relative_to(root.resolve()).as_posix()


def make_case_config(base, case, shared_budget):
    """Copy only sensor/controller settings, never renderer geometry or region IDs."""
    if set(base) != CONFIG_KEYS:
        raise ValueError(f"Unexpected or missing base experiment fields: {set(base) ^ CONFIG_KEYS}")
    if set(case) - CASE_KEYS:
        raise ValueError(f"Case overrides are forbidden: {set(case) - CASE_KEYS}")
    if not re.fullmatch(r"[a-z][a-z0-9_]*", case["id"]):
        raise ValueError("Case IDs must be simple lowercase identifiers")
    if case.get("split") != "development":
        raise ValueError("This exploratory suite must use development cases")
    if set(shared_budget) != {"max_frames", "max_stations"} or any(
        type(value) is not int or value <= 0 for value in shared_budget.values()
    ):
        raise ValueError("Supply common positive integer frame/station budgets")
    position = case["start_position"]
    if len(position) != 3 or any(not math.isfinite(float(value)) for value in position):
        raise ValueError("Start position must contain three finite coordinates")
    if position[2] != 0 or not math.isfinite(float(case["initial_body_yaw_deg"])):
        raise ValueError("The existing controller assumes a level floor at z=0")
    config = copy.deepcopy(base)
    config.update(shared_budget)
    config.update(
        name=f"space_suite_{case['id']}",
        start_position=list(position),
        initial_body_yaw_deg=case["initial_body_yaw_deg"],
    )
    return config


def build_provenance(suite_path, root=ROOT):
    suite_path = root / local_path(suite_path, root)
    suite = read_json(suite_path)
    if suite.get("schema_version") != 1 or not suite.get("cases"):
        raise ValueError("Expected a nonempty version 1 space suite")
    base_path = root / local_path(suite["base_experiment"], root)
    base = read_json(base_path)
    if suite["checkpoint"] != base["checkpoint"]:
        raise ValueError("All cases must retain the base experiment's frozen checkpoint")
    inputs = {local_path(suite_path, root), local_path(base_path, root)}
    inputs.add(local_path(suite["checkpoint"], root))
    inputs.update(PIPELINE_FILES)
    inputs.update(path.relative_to(root).as_posix() for path in (root / "src").rglob("*.py"))
    configs = {}
    for case in suite["cases"]:
        if case["id"] in configs:
            raise ValueError("Case IDs must be unique")
        configs[case["id"]] = make_case_config(base, case, suite["shared_budget"])
        inputs.add(local_path(case["building"], root))
    return {
        "schema_version": 1,
        "suite": suite,
        "case_configs": configs,
        "input_sha256": {name: sha256(root / name) for name in sorted(inputs)},
    }


def verify_inputs(provenance, root=ROOT):
    for name, digest in provenance["input_sha256"].items():
        if not (root / name).is_file() or sha256(root / name) != digest:
            raise FrozenInputError(f"Frozen suite input changed: {name}")


def prepare_output(output, provenance, resume=False):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        if not resume:
            raise FileExistsError("Suite output must be fresh; use --resume for verified cases")
        if not (output / "suite.json").is_file() or read_json(output / "suite.json") != provenance:
            raise ValueError("Resume requires exactly matching frozen suite provenance")
    else:
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(output / "suite.json", provenance)


def verify_completed(case_dir):
    status_path = case_dir / "status.json"
    if not status_path.exists() or read_json(status_path).get("status") != "complete":
        raise FileExistsError(
            f"Partial/failed case preserved at {case_dir}; choose a fresh output for a new attempt"
        )
    receipt = read_json(case_dir / "receipt.json")
    if not receipt.get("output_sha256"):
        raise ValueError("Completed case has no verifiable output receipt")
    for name, digest in receipt["output_sha256"].items():
        path = case_dir / local_path(name, case_dir)
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Completed case artifact changed: {path}")


def terminate_group(process, timeout_s=10):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=timeout_s)


def run_command(command, log_path, *, cpu=False, timeout_s=3600):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    if cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [str(arg) for arg in command],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return_code = process.wait(timeout=timeout_s)
        finally:
            terminate_group(process)
    if return_code:
        raise RuntimeError(f"Command exited {return_code}; inspect {log_path}")


class Renderer:
    """Own exactly one warm renderer; STOP, then exact-CID fallback, never a broad kill."""

    def __init__(self, case_dir, building, max_frames, *, ready_timeout_s=240, stop_timeout_s=90):
        self.case_dir = case_dir
        self.capture = case_dir / "capture"
        self.requests = case_dir / "requests"
        self.building, self.max_frames = building, max_frames
        self.ready_timeout_s, self.stop_timeout_s = ready_timeout_s, stop_timeout_s
        self.cidfile = case_dir / "renderer.cid"
        self.process = self.log = None

    def __enter__(self):
        self.capture.mkdir()
        self.requests.mkdir()
        self.log = (self.case_dir / "renderer.log").open("w")
        env = os.environ.copy()
        env["ROOMGRAPH_CONTAINER_CIDFILE"] = str(self.cidfile.resolve())
        command = [
            "./installation/run_python.sh",
            "scripts/render_building.py",
            "--config",
            local_path(self.building),
            "--output",
            local_path(self.capture),
            "--requests",
            local_path(self.requests),
            "--max-frames",
            str(self.max_frames),
        ]
        try:
            self.process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdout=self.log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            started = time.monotonic()
            while time.monotonic() - started < self.ready_timeout_s:
                if self.process.poll() is not None:
                    raise RuntimeError(f"Renderer exited during startup; inspect {self.log.name}")
                status = self.capture / "status.json"
                if status.exists():
                    value = read_json(status)
                    if value.get("status") == "ready":
                        return self
                    if value.get("status") == "error":
                        raise RuntimeError(f"Renderer startup failed: {value.get('error')}")
                time.sleep(0.25)
            raise TimeoutError(f"Renderer readiness timed out; inspect {self.log.name}")
        except BaseException:
            self.close()
            raise

    def close(self):
        self.requests.mkdir(exist_ok=True)
        (self.requests / "STOP").touch()
        forced = False
        try:
            if self.process is not None:
                try:
                    self.process.wait(timeout=self.stop_timeout_s)
                except subprocess.TimeoutExpired:
                    forced = True
                # Even an exited Docker client could have left its container alive.
                if self.cidfile.exists():
                    cid = self.cidfile.read_text().strip()
                    if not re.fullmatch(r"[0-9a-f]{12,64}", cid):
                        raise ValueError("Invalid renderer CID; refusing unrelated process cleanup")
                    inspected = subprocess.run(
                        ["docker", "inspect", "--format", "{{.State.Running}}", cid],
                        capture_output=True,
                        text=True,
                        timeout=15,
                    )
                    if inspected.returncode and not any(
                        text in inspected.stderr.lower()
                        for text in ("no such object", "no such container")
                    ):
                        raise RuntimeError(
                            "Could not determine whether the renderer is still alive"
                        )
                    if inspected.returncode == 0 and inspected.stdout.strip() == "true":
                        forced = True
                        stopped = subprocess.run(
                            ["docker", "stop", "--time", "10", cid],
                            capture_output=True,
                            text=True,
                            timeout=30,
                        )
                        if stopped.returncode:
                            subprocess.run(
                                ["docker", "kill", cid],
                                capture_output=True,
                                timeout=15,
                                check=True,
                            )
        except Exception as error:
            raise RendererCleanupError(f"Renderer cleanup failed: {error}") from error
        finally:
            if self.process is not None:
                terminate_group(self.process)
            if self.log is not None:
                self.log.close()
            atomic_json(
                self.case_dir / "renderer_stop.json",
                {
                    "stop_requested": True,
                    "forced_cleanup": forced,
                    "return_code": None if self.process is None else self.process.returncode,
                },
            )

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


def run_case(case, config, case_dir, provenance, *, skip_report=False, ready_timeout_s=240):
    if case_dir.exists() and any(case_dir.iterdir()):
        raise FileExistsError(f"Case directory must be fresh: {case_dir}")
    case_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(case_dir / "experiment_config.json", config)
    started = time.perf_counter()

    def stage(name):
        print(f"[{case['id']}] {name}", flush=True)
        atomic_json(case_dir / "status.json", {"status": "running", "stage": name})

    try:
        verify_inputs(provenance)
        stage("renderer_startup")
        with Renderer(
            case_dir,
            case["building"],
            max(1000, config["max_frames"] + 1),
            ready_timeout_s=ready_timeout_s,
        ):
            stage("acquisition")
            run_command(
                [
                    sys.executable,
                    "scripts/run_multiroom.py",
                    "--config",
                    local_path(case_dir / "experiment_config.json"),
                    "--capture",
                    local_path(case_dir / "capture"),
                    "--requests",
                    local_path(case_dir / "requests"),
                    "--output",
                    local_path(case_dir / "run"),
                ],
                case_dir / "acquisition.log",
            )
        verify_inputs(provenance)
        results = local_path(case_dir / "run/experiment.json")
        reference = local_path(case_dir / "capture/reference.json")
        report = local_path(case_dir / "report")
        commands = [
            (
                "evaluation",
                ["scripts/evaluate_multiroom.py", "--results", results, "--reference", reference],
            ),
            (
                "replay",
                [
                    "scripts/check_multiroom_replay.py",
                    "--results",
                    results,
                    "--output",
                    local_path(case_dir / "run/replay_check.json"),
                ],
            ),
            ("observed_mesh", ["scripts/export_observed_mesh.py", "--results", results]),
        ]
        if not skip_report:
            commands.extend(
                [
                    (
                        "report",
                        [
                            "scripts/make_multiroom_report.py",
                            "--results",
                            results,
                            "--reference",
                            reference,
                            "--output",
                            report,
                        ],
                    ),
                    (
                        "topology",
                        [
                            "scripts/make_multiroom_topology.py",
                            "--run",
                            local_path(case_dir / "run"),
                            "--output",
                            report,
                        ],
                    ),
                    (
                        "refresh_page",
                        ["scripts/make_multiroom_report.py", "--output", report, "--refresh-page"],
                    ),
                ]
            )
        for name, command in commands:
            stage(name)
            run_command([sys.executable, *command], case_dir / f"{name}.log", cpu=True)
        if not skip_report:
            stage("video")
            run_command(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-framerate",
                    "8",
                    "-i",
                    str(case_dir / "report/replay/frame_%04d.webp"),
                    "-c:v",
                    "libopenh264",
                    "-b:v",
                    "1400k",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    "-an",
                    str(case_dir / "report/replay.mp4"),
                ],
                case_dir / "video.log",
                cpu=True,
            )
            if (case_dir / "report/replay.mp4").stat().st_size >= 10_000_000:
                raise ValueError("Selected public video exceeds the 10 MB budget")
        verify_inputs(provenance)
        artifacts = [
            "experiment_config.json",
            "capture/manifest.json",
            "capture/reference.json",
            "run/experiment.json",
            "run/evaluation.json",
            "run/replay_check.json",
            "run/observed_walls.json",
            "run/observed_walls.obj",
            "run/topology.json",
        ]
        if not skip_report:
            artifacts.extend(
                [
                    "report/summary.json",
                    "report/report.html",
                    "report/topology.json",
                    "report/replay.mp4",
                    "report/poster.jpg",
                ]
            )
        if read_json(case_dir / "run/replay_check.json").get("status") != "pass":
            raise ValueError("Replay did not pass")
        atomic_json(
            case_dir / "receipt.json",
            {
                "output_sha256": {name: sha256(case_dir / name) for name in artifacts},
                "report_generated": not skip_report,
            },
        )
        atomic_json(
            case_dir / "status.json",
            {
                "status": "complete",
                "stage": "complete",
                "wall_seconds": time.perf_counter() - started,
                "report_generated": not skip_report,
            },
        )
    except BaseException as error:
        atomic_json(
            case_dir / "status.json",
            {
                "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                "stage": read_json(case_dir / "status.json").get("stage")
                if (case_dir / "status.json").exists()
                else "preflight",
                "error": f"{type(error).__name__}: {error}",
                "wall_seconds": time.perf_counter() - started,
            },
        )
        raise


def suite_status(output, cases, status):
    entries = {}
    for case in cases:
        path = output / case["id"] / "status.json"
        entries[case["id"]] = read_json(path) if path.exists() else {"status": "pending"}
    if status == "complete" and any(item["status"] != "complete" for item in entries.values()):
        status = "partial"
    atomic_json(output / "status.json", {"status": status, "cases": entries})


def execute_cases(provenance, output, selected, *, skip_report=False, ready_timeout_s=240):
    """Continue independent failures, but halt on interruption or compromised isolation."""
    cases = provenance["suite"]["cases"]
    failures = []
    for case in cases:
        if case["id"] not in selected:
            continue
        verify_inputs(provenance)
        case_dir = output / case["id"]
        try:
            if case_dir.exists() and any(case_dir.iterdir()):
                verify_completed(case_dir)
                print(f"[{case['id']}] verified complete; skipping", flush=True)
                continue
            run_case(
                case,
                provenance["case_configs"][case["id"]],
                case_dir,
                provenance,
                skip_report=skip_report,
                ready_timeout_s=ready_timeout_s,
            )
        except (FrozenInputError, RendererCleanupError):
            raise
        except Exception as error:
            verify_inputs(provenance)
            failures.append(case["id"])
            print(f"[{case['id']}] preserved failure: {error}", flush=True)
        suite_status(output, cases, "running")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", type=Path, default=Path("configs/experiments/space_suite_v1.json")
    )
    parser.add_argument("--output", type=Path, default=Path("vis/space_suite/v1"))
    parser.add_argument("--case", action="append", help="Select a case; may be repeated")
    parser.add_argument("--skip-report", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip verified completed cases")
    parser.add_argument("--ready-timeout-s", type=float, default=240)
    args = parser.parse_args()
    if not math.isfinite(args.ready_timeout_s) or args.ready_timeout_s <= 0:
        parser.error("Renderer readiness timeout must be finite and positive")
    output = ROOT / local_path(args.output)
    provenance = build_provenance(args.suite)
    cases = provenance["suite"]["cases"]
    selected = set(args.case or [case["id"] for case in cases])
    if selected - {case["id"] for case in cases}:
        parser.error("Unknown case ID")
    prepare_output(output, provenance, args.resume)

    def interrupted(_signum, _frame):
        raise KeyboardInterrupt("Suite interrupted; stopping the current renderer")

    signal.signal(signal.SIGTERM, interrupted)
    suite_status(output, cases, "running")
    try:
        failures = execute_cases(
            provenance,
            output,
            selected,
            skip_report=args.skip_report,
            ready_timeout_s=args.ready_timeout_s,
        )
    except BaseException as error:
        suite_status(
            output, cases, "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        )
        raise
    suite_status(output, cases, "failed" if failures else "complete")
    print(json.dumps(read_json(output / "status.json"), indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
