"""Replay the versioned commitment policy using archived acquired inputs only."""

import check_multiroom_replay as baseline

from roomgraph.policy_runtime import (
    policy_bindings,
    policy_specification,
    require_decision_trace,
    validate_replay_manifest,
)

BASELINE_REPLAY = baseline.replay


def replay(results, *, atol=0):
    validate_replay_manifest(results)
    with policy_bindings(baseline) as policy:
        report = BASELINE_REPLAY(results, atol=atol)
        require_decision_trace(policy, results)
    return {
        **report,
        "exploration_policy": {
            **policy_specification(),
            "decisions_checked": len(policy.decisions),
        },
        "verified": [*report["verified"], "bounded frontier commitment decision trace"],
    }


def main():
    original = baseline.replay
    baseline.replay = replay
    try:
        baseline.main()
    finally:
        baseline.replay = original


if __name__ == "__main__":
    main()
