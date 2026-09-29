"""Render-only replay of a completed policy using acceleration-limited head motion."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from roomgraph.active_view import HeadMotionLimits


def _axis_position(distance, elapsed, speed, acceleration):
    if distance == 0:
        return np.zeros_like(elapsed, dtype=float)
    peak_speed = min(speed, np.sqrt(distance * acceleration))
    ramp_time = peak_speed / acceleration
    cruise_time = max(0, distance / peak_speed - ramp_time)
    duration = 2 * ramp_time + cruise_time
    time = np.clip(elapsed, 0, duration)
    return np.where(
        time < ramp_time,
        0.5 * acceleration * time**2,
        np.where(
            time < ramp_time + cruise_time,
            0.5 * acceleration * ramp_time**2 + peak_speed * (time - ramp_time),
            distance - 0.5 * acceleration * (duration - time) ** 2,
        ),
    )


def interpolate_head(start, target, time_s, limits):
    """Joint angles at elapsed seconds; each axis finishes independently then holds.

    ``time_s`` may be a scalar or array. The final axis indexes yaw and pitch.
    Every move starts and ends at rest, matching ``HeadMotionLimits.duration``.
    """
    limits.validate(start)
    limits.validate(target)
    elapsed = np.asarray(time_s, dtype=float)
    if not np.isfinite(elapsed).all() or np.any(elapsed < 0):
        raise ValueError("Elapsed time must be finite and nonnegative")
    delta = np.asarray(target, dtype=float) - start
    axes = [
        _axis_position(abs(distance), elapsed, speed, acceleration) * np.sign(distance) + initial
        for initial, distance, speed, acceleration in zip(
            start, delta, limits.speed_deg_s, limits.accel_deg_s2, strict=True
        )
    ]
    return np.stack(axes, axis=-1)


def build_timeline(trace, budget_s, fps, limits):
    """Sample the executed decisions, including settling and unused budget hold."""
    if not trace or not np.isfinite([budget_s, fps]).all() or min(budget_s, fps) <= 0:
        raise ValueError("A nonempty trace, positive budget and positive frame rate are required")
    timestamps = np.asarray([row["t_s"] for row in trace], dtype=float)
    if (
        not np.isfinite(timestamps).all()
        or abs(timestamps[0]) > 1e-8
        or np.any(np.diff(timestamps) <= 0)
        or timestamps[-1] > budget_s + 1e-8
    ):
        raise ValueError("Trace timestamps must increase from zero and remain within budget")
    joints = [(row["yaw_deg"], row["pitch_deg"]) for row in trace]
    for angles in joints:
        limits.validate(angles)
    for index in range(1, len(trace)):
        expected = limits.duration(joints[index - 1], joints[index])
        if not np.isclose(timestamps[index] - timestamps[index - 1], expected, atol=1e-7):
            raise ValueError("Trace timestamps do not match the configured motion limits")
    times = np.arange(int(np.floor(budget_s * fps)) + 1, dtype=float) / fps
    if not np.isclose(times[-1], budget_s, atol=1e-10):
        times = np.r_[times, budget_s]
    else:
        times[-1] = budget_s
    frames = []
    for time in times:
        acquired = min(int(np.searchsorted(timestamps, time, side="right")) - 1, len(trace) - 1)
        target = min(acquired + 1, len(trace) - 1)
        if target == acquired:
            angles, phase = joints[acquired], "holding"
        else:
            elapsed = float(time - timestamps[acquired])
            angles = interpolate_head(joints[acquired], joints[target], elapsed, limits)
            movement_time = timestamps[target] - timestamps[acquired] - limits.settle_s
            phase = "moving" if elapsed < movement_time else "settling"
        frames.append(
            {
                "time_s": float(time),
                "yaw_deg": float(angles[0]),
                "pitch_deg": float(angles[1]),
                "last_acquired_view_id": trace[acquired]["view_id"],
                "target_view_id": trace[target]["view_id"],
                "target_time_s": float(timestamps[target]),
                "phase": phase,
                "source_policy": "active",
                "visualization_only": True,
            }
        )
    return frames


def prepare(results, manifest, fps=5):
    """Copy the furnished scene and replace its camera bank with dense replay views."""
    if results.get("status") != "complete":
        raise ValueError("Dense visualization requires a completed experiment")
    config = results["config"]
    active = [policy for policy in results["policies"] if policy["name"] == "active"]
    if len(active) != 1:
        raise ValueError("Expected exactly one completed active policy")
    if manifest["scene_config"]["name"] != config["capture_name"]:
        raise ValueError("Capture scene does not match the completed experiment")
    limits = HeadMotionLimits(**config["motion_limits"])
    frames = build_timeline(active[0]["trace"], config["action_budget_s"], fps, limits)
    scene = copy.deepcopy(manifest["scene_config"])
    scene.update(
        name=config["capture_name"].removesuffix("_head") + "_motion",
        title="Continuous active head-camera replay",
        description=(
            "Visualization-only rendering of the frozen active policy. Extra intermediate "
            "frames are excluded from action selection, mapping and benchmark metrics."
        ),
    )
    scene["render"].update(config["render"])
    scene["cameras"] = [
        {
            "id": f"M{index:03d}",
            "focal_length_mm": config["focal_length_mm"],
            "robot_base": copy.deepcopy(config["robot_base"]),
            "head": {"yaw_deg": frame["yaw_deg"], "pitch_deg": frame["pitch_deg"]},
            "visualization": frame,
        }
        for index, frame in enumerate(frames)
    ]
    scene["head_motion_visualization"] = {
        "fps": fps,
        "duration_s": config["action_budget_s"],
        "source_policy": "active",
        "motion_profile": "independent simultaneous rest-to-rest trapezoid or triangle per joint",
        "visualization_only": True,
        "excluded_from_policy_and_metrics": True,
        "timeline": frames,
        "head_limits_note": "Illustrative simulation assumptions, not official 1X specifications",
    }
    return scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, default=Path("vis/active_head/run_v1/experiment.json")
    )
    parser.add_argument("--capture", type=Path, default=Path("vis/active_head/capture"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/active-head-motion.json"))
    parser.add_argument("--fps", type=float, default=5)
    args = parser.parse_args()
    results = json.loads(args.results.read_text())
    manifest_path = args.capture / results["config"]["capture_name"] / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    if results.get("capture_manifest_sha256") != manifest_hash:
        raise ValueError("Capture manifest checksum differs from the completed experiment")
    scene = prepare(results, json.loads(manifest_bytes), args.fps)
    scene["head_motion_visualization"].update(
        experiment_path=str(args.results.resolve()),
        experiment_sha256=hashlib.sha256(args.results.read_bytes()).hexdigest(),
        capture_manifest_sha256=manifest_hash,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scene, indent=2) + "\n")
    print(f"{len(scene['cameras'])} visualization-only motion frames: {args.output}")


if __name__ == "__main__":
    main()
