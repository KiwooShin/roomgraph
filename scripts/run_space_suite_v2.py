"""Compare bounded commitment on the same four furnished development cases.

Reuse the frozen suite's renderer ownership, cleanup, evaluation and reporting.
Only the acquisition and replay entrypoints change; baseline files and outputs
remain available unchanged.
"""

import sys
from contextlib import contextmanager

import run_space_suite as baseline

from roomgraph.policy_runtime import VERSIONED_SOURCE_FILES, policy_specification

COMMAND_ADAPTERS = {
    "scripts/run_multiroom.py": "scripts/run_multiroom_v2.py",
    "scripts/check_multiroom_replay.py": "scripts/check_multiroom_replay_v2.py",
}
DEFAULT_SUITE = "configs/experiments/space_suite_v2.json"
DEFAULT_OUTPUT = "vis/space_suite/v2"


def adapt_command(command):
    """Redirect exact Python entrypoints, never incidental argument strings."""
    result = list(command)
    if len(result) > 1 and str(result[0]) == sys.executable:
        result[1] = COMMAND_ADAPTERS.get(str(result[1]), result[1])
    return result


def validate_suite(suite, reference):
    if suite.get("exploration_policy") != policy_specification():
        raise ValueError("Suite must declare the fixed versioned policy ID/settings")
    for key in ("base_experiment", "checkpoint", "shared_budget", "cases"):
        if suite.get(key) != reference[key]:
            raise ValueError(f"Policy comparison must preserve baseline {key}")
    if suite.get("name") == reference["name"]:
        raise ValueError("Policy comparison needs a distinct suite name")


@contextmanager
def suite_bindings():
    original_command = baseline.run_command
    original_provenance = baseline.build_provenance
    original_files = baseline.PIPELINE_FILES

    def run_command(command, *args, **kwargs):
        return original_command(adapt_command(command), *args, **kwargs)

    def build_provenance(suite_path, root=baseline.ROOT):
        suite = baseline.read_json(root / baseline.local_path(suite_path, root))
        reference = baseline.read_json(root / "configs/experiments/space_suite_v1.json")
        validate_suite(suite, reference)
        return original_provenance(suite_path, root)

    baseline.run_command = run_command
    baseline.build_provenance = build_provenance
    baseline.PIPELINE_FILES = tuple(dict.fromkeys((*original_files, *VERSIONED_SOURCE_FILES)))
    try:
        yield
    finally:
        baseline.run_command = original_command
        baseline.build_provenance = original_provenance
        baseline.PIPELINE_FILES = original_files


def main():
    original_argv = sys.argv
    sys.argv = [sys.argv[0], "--suite", DEFAULT_SUITE, "--output", DEFAULT_OUTPUT, *sys.argv[1:]]
    try:
        with suite_bindings():
            baseline.main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
