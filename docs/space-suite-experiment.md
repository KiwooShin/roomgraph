# Multiple furnished spaces

This development suite runs the same head-camera exploration and reconstruction
pipeline on four different buildings. It tests how layout and furnishing change
coverage and failure modes. The cases are fixed before acquisition and use the
existing frozen edge checkpoint; there is no new training or per-case policy
tuning. These examples are not a held-out generalization benchmark.

| Case | Layout | Area | Reference rooms |
| --- | --- | ---: | ---: |
| `residential_branch` | Living room, kitchen, bedroom and study around a branching corridor | 80 m² | 4 |
| `loop_workspaces` | Studio, dining room, lounge and workshop connected in a cycle | 72 m² | 4 |
| `open_office` | A 54 m² open office with three smaller side rooms | 90 m² | 4 |
| `compact_apartment` | Four furnished rooms off a 1.7 m corridor, with 1.4 m openings | 56 m² | 4 |

The editable [building descriptions](../configs/buildings/cases/) contain physical
walls, portal openings, furniture recipes, lighting, render settings and initial
poses. Reference region boundaries follow physical partitions; desk clusters in
the open office do not create invented room edges. Portals are open passages,
without articulated door leaves. The humanoid is the original articulated proxy,
not a calibrated model of a commercial robot.

## Fixed protocol

The [suite configuration](../configs/experiments/space_suite_v1.json) supplies a
common limit of 300 acquired images and 32 body stations. All cases use 384 × 256
images, 16 path-tracing samples per pixel, the same trained checkpoint, the same
frontier policy, and the same mapping and motion assumptions. The initial pose
and physical scene vary. Total available architecture and navigable distances
differ, so equal image budgets need not yield equal coverage.

The controller receives only acquired RGB, ideal optical-axis depth and known
camera poses. RGB drives learned structural-edge predictions; depth supports
occupancy, colored surface fusion and lifting observed predicted edges into 3D.
This is **sensor-assisted RGB-D reconstruction**, not monocular reconstruction.
Building labels, reference geometry and future observations remain outside the
planner. The [connected-room protocol](multiroom-experiment.md) describes the
self-body mask, observed-free movement checks and map integration.

Every suite records hashes of the checkpoint, case configurations and pipeline
sources before its first capture. Inputs are checked between cases. Each case
gets isolated capture, request, run and report directories. A warm Isaac renderer
serves sequential observations and stops before replay and reporting; only one
case uses the GPU at a time. Container cleanup targets its recorded container ID.

## Run and inspect

Install Isaac and prepare the existing trained checkpoint and furnishing assets
as described in the repository README and [perception guide](perception.md).
Then run from the repository root with the CUDA-enabled host Python:

```bash
PYTHONPATH=src python scripts/run_space_suite.py \
  --suite configs/experiments/space_suite_v1.json --output vis/space_suite/v1
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' python scripts/make_space_suite_report.py \
  --suite configs/experiments/space_suite_v1.json \
  --root vis/space_suite/v1 --output vis/space_suite/report
```

Use a fresh output directory for a new experiment. `--case ID` selects particular
cases; `--resume` reuses completed cases only after verifying their provenance
and output receipts. Partial or failed case directories remain intact for
inspection and are not silently overwritten.

The aggregate report links to each case's moving-camera replay, furnished
top-down view with camera directions, RGB/edge overlays, partial occupancy map,
interactive observed point cloud, and supported wall patches. Full renders,
depth, predictions, maps and reports stay in ignored `vis/`. Only selected media
and compact metadata are copied into the separate public website repository.

## Measurements and interpretation

Report every declared case, including incomplete and failed cases. For completed
acquisitions measure reference rooms entered, portal crossings, architectural
edge visibility, learned 3D edge precision/completeness at 10 cm, and their typed
variants. Visibility means the camera saw reference structure; completeness means
the fused prediction recovered it within the distance tolerance. Neither room
entry nor a frontier stop establishes whole-building reconstruction.
Count distinct crossed portals against the total physical openings separately.
A loop-shaped building can be visited through a spanning path without crossing
the remaining connection. Known poses also remove the SLAM loop-closure problem;
these experiments do not evaluate it.

Summarize both the equal-building macro average and denominator-weighted values.
Weight visibility and completeness by reference edge length, and precision by
the number of predicted edge voxels. State the number of measured buildings and
retain unavailable cases as missing results, not invented zeros. Curves compare
visibility against travel distance and simulated head/body action time.

Independent CPU replay checks archived input hashes, map arrays, head targets,
frontier choices, movement paths and the action clock. Evaluate movement against
the map available before each move, then separately audit continuous circular
footprints against architecture and furniture recipe bounds. These bounds do not
certify downloaded meshes, whole-body collision avoidance or humanoid walking.

Record acquisition wall time and stage timings separately from simulated action
time and full orchestration time, which includes startup, evaluation and media
generation. The fixed checkpoint avoids retraining cost; these cases first reveal
where broader training or improved observation allocation would help. Any later
policy change needs a new frozen suite run, preserving the original results.

The renderer and model share one GPU sequentially: the server waits between
requested observations while the host runs inference. Offline replay and report
generation run on the CPU after the renderer stops. Model-forward timing alone
is not pipeline throughput; retain acquisition, serialization, fusion and startup
costs when deciding what to optimize next. In particular, a faster network cannot
remove the cost of redundant rendered head sweeps.

## Measured development results

The frozen `vis/space_suite/v1` run acquired 300 images at 30 stations in each
building, reusing the existing head-camera checkpoint. Every case stopped at the
image budget. The same model and policy entered 13 of 16 reference rooms; only
two of four buildings had every room entered.

| Case | Rooms entered | Edges visible | 3D precision | 3D completeness | Typed completeness |
| --- | ---: | ---: | ---: | ---: | ---: |
| Branching home | 4 / 4 | 84.17% | 69.15% | 44.73% | 29.05% |
| Loop workspaces | 4 / 4 | 84.88% | 79.50% | 43.27% | 32.37% |
| Open office | 2 / 4 | 82.02% | 42.97% | 42.76% | 26.42% |
| Compact apartment | 3 / 4 | 74.15% | 64.25% | 45.13% | 31.13% |

All reconstruction scores in the table use a 10 cm tolerance. The office never
entered the break room or quiet office; the apartment never entered its kitchen.
Views through doorways still reveal some structure in these spaces, explaining
why high visibility can coexist with incomplete room visitation. The loop case
crossed only three of its four physical connections. These outcomes remain in
the public report rather than being removed or repaired by per-case tuning.

Equal-building macro visibility was 81.30%, precision 63.97% and completeness
43.98%. Reference-length-weighted visibility/completeness were 81.31% / 44.03%;
predicted-voxel-weighted precision was 61.87%. The report records the underlying
denominators rather than averaging percentages without their measurement scope.

All 116 executed movements passed their pre-move observed-safe map checks. The
independent continuous 25 cm disk audits found no architecture or furniture-proxy
intersections. The smallest remaining furniture-proxy clearance was 4 cm in the
loop case; this is not a full-body collision or locomotion validation.

Acquisition took 282.39, 286.98, 288.39 and 278.58 seconds respectively, separate
from simulated motion time and offline reporting. Median capture times were
660–679 ms per image, compared with 7.55–7.69 ms for model inference and about
80–82 ms for self filtering and map fusion. Rendering dominated measured
acquisition; the first case spent 71.1% of its acquisition wall time in capture
and 1.1% in the measured network stage. Profile serialization and reduce redundant
observations before assuming further network acceleration is the main benefit.
Total acquisition wall time was 18.94 minutes. The four complete pipelines took
24.74 minutes including renderer startup, evaluation, replay and media generation.

Visual inspection confirmed the physical partitions, open passages, loaded
assets and articulated head views in all four cases. A local furnishing defect
remains in the frozen apartment: plant foliage intersects the sofa's right arm
and cushion in frames `F0004` and `F0036`. The acquired imagery is retained.
Future furnishing validation should use imported-mesh bounds and placement
constraints, beyond the procedural-proxy checks used here.

The source passes 149 CPU unit tests and Ruff; the website publisher has six
additional tests for provenance and selected-media validation. All four independent
CPU replays passed, reproducing 1,320 map arrays with zero tolerance and verifying
the recorded head targets, frontier decisions, paths and final topology. This
reuses archived predictions and does not rerun network inference or rendering.
Full local reports
are in `vis/space_suite/report/report.html` and each case's `report/report.html`.
The [public comparison](https://kiwooshin.github.io/roomgraph-spaces/) contains
selected media, interactive observed maps and denominator-aware aggregates.

## Follow-up: persistent exploration goals

The first suite exposed goal changes before arrival, despite paths that remained
safe in the acquired maps. In the office, station 7 selected a break-room frontier
5.4 m away and station 8 abandoned it after one 0.85 m movement. Its corresponding
frontier had shifted only 0.30 m and was still reachable. In the apartment,
station 16 returned to the exact position of station 10 and acquired ten images
that added no new surface voxels. The apartment spent 118 of 300 images at stations
adding fewer than 500 surface voxels each. These are recorded diagnostic findings,
not a counterfactual estimate of a different policy's performance.

The follow-up changes only frontier-goal persistence. A selected goal may remain
active across several short movements, while its path is recalculated against the
latest observed map. Arrival, blocked paths, loss of a spatially matching frontier,
stalled progress or a bounded commitment duration release it. The trace calls a
lost frontier match `resolved`; this does not certify a completed region. New
goals use the original frontier scoring rule. Head sweeps, mapping, footprint,
checkpoint, rendering,
starts and image/station budgets remain unchanged. No supplied room labels or
door destinations enter this policy.

Versioned runtime, replay and suite adapters leave the published baseline's
source files and captures intact. A fresh suite goes under `vis/space_suite/v2`.
The policy settings and sources are frozen before acquisition; independent replay
must reconstruct both its stateful decisions and all observed maps. Compare every
case, including regressions, against the recorded first suite. This is a refinement
tuned from these development cases, not an unseen-building generalization test.

This comparison uses fresh captures with the same renderer settings, not identical
RGB tensors. For the branching home's initial 12 views, requests and depth-file
hashes matched v1 exactly, while RGB mean absolute difference was 0.31 intensity
levels on the 0–255 scale. Model probability mean absolute difference was 0.000323.
Within-run replay uses archived probabilities; between-policy model scores can
also reflect this small rendering variation. A single run per layout and policy
does not estimate statistical significance.

The initial 32 requested views also matched between the two branching-home runs,
and the first three station occupancy arrays were identical. The first different
goal choice occurred on these matching acquired maps. This supports attributing
the initial planning divergence to the policy change, while the reconstruction
scores still include rendering and inference variation. The full image/depth
audit remains local at
`vis/space_suite/v2/residential_branch_initial_view_audit.json`.

The [v2 suite declaration](../configs/experiments/space_suite_v2.json) records the
fixed policy settings. Run the follow-up and paired report with:

```bash
PYTHONPATH=src python scripts/run_space_suite_v2.py
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' python scripts/make_space_suite_report.py \
  --suite configs/experiments/space_suite_v2.json \
  --root vis/space_suite/v2 --output vis/space_suite/report_v2
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' python scripts/compare_space_policies.py \
  --baseline-report vis/space_suite/report --baseline-root vis/space_suite/v1 \
  --candidate-report vis/space_suite/report_v2 --candidate-root vis/space_suite/v2 \
  --output vis/space_suite/policy_comparison
```

### Follow-up results and decision

The candidate completed all four 300-image runs, still entering 13 of 16 rooms.
The same office break room, quiet office and apartment kitchen remained
unentered. Bounded goal persistence alone did not fix the missed-room problem,
so **v1 remains the default**. Preserve v2 as a measured experimental candidate.

Each cell below shows v1 → v2; reconstruction uses a 10 cm tolerance.

| Case | Rooms entered | Edges visible | 3D precision | 3D completeness |
| --- | ---: | ---: | ---: | ---: |
| Branching home | 4/4 → 4/4 | 84.17% → 81.96% | 69.15% → 64.95% | 44.73% → 43.43% |
| Loop workspaces | 4/4 → 4/4 | 84.88% → 85.45% | 79.50% → 75.42% | 43.27% → 44.73% |
| Open office | 2/4 → 2/4 | 82.02% → 73.22% | 42.97% → 45.65% | 42.76% → 37.17% |
| Compact apartment | 3/4 → 3/4 | 74.15% → 73.18% | 64.25% → 62.05% | 45.13% → 44.34% |
| Equal-building macro | 3.25/4 → 3.25/4 | 81.30% → 78.45% | 63.97% → 62.02% | 43.98% → 42.42% |

The loop layout gained 1.46 percentage points of completeness but lost 4.08
points of precision. The office lost 8.80 points of visibility and 5.60 points
of completeness. Macro completeness fell 1.56 points, with lower scores in
three of four layouts; macro typed completeness fell from 29.74% to 29.04%.
These results argue against adopting this candidate despite its reduced goal
switching. Better persistence is not, by itself, better selection of informative
viewpoints.

The recorded early-goal-switch proxy fell from 33 to 13, but frames acquired at
stations adding fewer than 500 surface voxels increased from 362 to 414. These
are descriptive diagnostics: a necessary switch can avoid a blocked route, and
a low-gain image can still improve confidence or edge predictions. The paired
report defines these thresholds and shows every case separately.

The office illustrates why unconditional persistence can hurt. Through station
7, both runs have identical occupancy arrays, frontier lists, camera requests
and depth hashes. From `(-1.35, -0.13)`, v1 selects the break-room frontier at
`(3.55, 0.15)`, whereas v2 retains a main-office goal at `(-2.35, 1.05)`.
At station 13, v2 retains a frontier with 5 gain cells while its local greedy
alternative has 37 cells. Later, stations 16–18 revisit nearby positions and
add only 189, 237 and 233 surface voxels. These observations explain concrete
detours; they do not prove that following the alternative would enter a room.
Room names here are assigned afterward by the evaluator.

A next hypothesis is to release a commitment when its remaining information
value becomes too small relative to other reachable frontiers and the remaining
image budget. This has not been implemented or validated. It should be tested
as a separate frozen candidate, retaining acquired-map-only decisions and all
four development outcomes, before any unseen-layout evaluation.

The candidate's acquisition took 19.03 minutes versus 18.94 minutes for v1,
with the same 1,200 images and 120 stations. Its total path was 97.06 m versus
98.48 m. This is not a demonstrated speedup: it acquired no fewer images and
produced less complete geometry overall. No model retraining was needed for
this planning experiment. Future efficiency work should measure reconstruction
quality per image and per elapsed second, rather than optimizing network latency
alone or assuming that fewer goal switches save useful observations.

All four v2 CPU replays passed: 1,320 map arrays matched exactly and all 120
stateful policy decisions were reproduced. All 45 frozen inputs and 56 receipted
outputs matched their SHA-256 records. The source passes 181 CPU unit tests and
Ruff; the publisher passes 13 tests. Full pipeline time, including rendering
startup, evaluation and media generation, was 24.87 minutes versus 24.74 minutes
for v1. The v2 apartment contact sheet still shows the frozen plant/sofa
intersection; its separate manual review is recorded locally rather than
inheriting the baseline's inspection claim.
