"""Run bounded frontier commitment with the unchanged RGB-D acquisition loop."""

import run_multiroom as baseline

from roomgraph.policy_runtime import policy_bindings


def main():
    with policy_bindings(baseline, capture_writes=True):
        baseline.main()


if __name__ == "__main__":
    main()
