"""Scoped adapters for the versioned frontier experiment and its exact replay.

The published runner remains byte-identical. These adapters replace only its
frontier selector, retain the existing scan schedule, and record every policy
decision alongside the source files that implemented it.
"""

import copy
import hashlib
from contextlib import contextmanager
from pathlib import Path

from roomgraph.frontier_commitment import DEFAULT_SETTINGS, POLICY_ID, FrontierCommitmentPolicy

ROOT = Path(__file__).resolve().parents[2]
VERSIONED_SOURCE_FILES = (
    "src/roomgraph/frontier_commitment.py",
    "src/roomgraph/policy_runtime.py",
    "scripts/run_multiroom_v2.py",
    "scripts/check_multiroom_replay_v2.py",
    "scripts/run_space_suite_v2.py",
    "scripts/check_multiroom_replay.py",
    "scripts/run_active_head.py",
    "configs/experiments/space_suite_v1.json",
    "configs/experiments/space_suite_v2.json",
)


def policy_specification():
    """Return an independent, serializable declaration of the fixed policy."""
    return {"id": POLICY_ID, "settings": copy.deepcopy(DEFAULT_SETTINGS)}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_sources(root=ROOT):
    """Capture all additional experiment dependencies before acquisition starts."""
    return {name: file_hash(Path(root) / name) for name in VERSIONED_SOURCE_FILES}


def verify_sources(expected, root=ROOT):
    for name in VERSIONED_SOURCE_FILES:
        if name not in expected:
            raise ValueError(f"Versioned policy source hash is missing: {name}")
        if file_hash(Path(root) / name) != expected[name]:
            raise ValueError(f"Versioned policy source SHA256 differs: {name}")


def validate_replay_manifest(results, root=ROOT):
    """Reject an unspecified or modified policy before loading observations."""
    policy = results.get("exploration_policy")
    if not isinstance(policy, dict) or set(policy) != {"id", "settings"}:
        raise ValueError("Replay requires the complete versioned policy manifest")
    if policy != policy_specification():
        raise ValueError("Replay policy ID/settings differ from this fixed experiment")
    decisions = results.get("policy_decisions")
    if not isinstance(decisions, list) or len(decisions) != len(results.get("stations", [])):
        raise ValueError("Policy decision trace must have one entry per acquired station")
    verify_sources(results.get("source_sha256", {}), root)


def require_decision_trace(policy, results):
    if policy.decisions != results["policy_decisions"]:
        raise ValueError("Replayed commitment decisions differ from the frozen trace")


@contextmanager
def policy_bindings(module, *, capture_writes=False, root=ROOT):
    """Bind one fresh policy for one synchronous run and always restore globals.

    ``observe_grid`` sees the same acquired map object subsequently updated by
    the unchanged sensor loop. It receives no room labels or renderer geometry.
    The process-local scope is intentionally not a concurrent execution API.
    """
    policy = FrontierCommitmentPolicy()
    original_scan = module.scan_targets
    original_choose = module.choose_frontier
    original_write = module.atomic_write_json if capture_writes else None
    source_hashes = freeze_sources(root) if capture_writes else None

    def scan_targets(grid, position, first_station):
        policy.observe_grid(grid, position)
        return original_scan(grid, position, first_station)

    def write_json(value, path):
        if Path(path).name == "experiment.json" and value.get("status") == "complete":
            verify_sources(source_hashes, root)
            value = {
                **value,
                "source_sha256": {**value["source_sha256"], **source_hashes},
                "exploration_policy": policy_specification(),
                "policy_decisions": copy.deepcopy(policy.decisions),
            }
            if len(policy.decisions) != len(value["stations"]):
                raise ValueError("Policy decisions do not match acquired station count")
        return original_write(value, path)

    module.scan_targets = scan_targets
    module.choose_frontier = policy.choose
    if capture_writes:
        module.atomic_write_json = write_json
    try:
        yield policy
    finally:
        module.scan_targets = original_scan
        module.choose_frontier = original_choose
        if capture_writes:
            module.atomic_write_json = original_write
