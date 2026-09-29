# Connected-room exploration with a head camera

This experiment starts with one supplied robot pose and an empty map. The robot
scans the starting room, chooses reachable frontiers of unknown space, moves a
short distance through observed free space, scans again, and accumulates a shared
3D map. Three furnished rooms connect through real openings into a corridor.

The sensor setup deliberately extends the previous RGB-only room-fitting pilot:
**ideal optical-axis depth and known metric poses are supplied**. RGB feeds the
existing frozen EdgeNet model. Acquired depth supports navigation and places
visible predicted edges in 3D. This isolates exploration and map integration; it
does not establish monocular SLAM, learned depth, humanoid walking control, or
general reconstruction of an unseen building.

## Scene and acquisition

[`connected_three_rooms.json`](../configs/buildings/connected_three_rooms.json)
describes an office, living room, bedroom, and corridor, including individual
furnishings and wall openings. The wall builder splits solid walls around each
portal and retains its overhead lintel. Doorways are 1.6 m wide; no door leaf is
present. Open passages are therefore real render geometry, not a painted door
on a solid wall. The camera and original humanoid proxy use the articulated head
rig from the [head-motion experiment](active-head-experiment.md).

The warm Isaac server receives one atomic pose request at a time and returns
only that RGB/depth frame and calibration. The controller's `FileCamera` boundary
whitelists these fields, validates requested versus returned calibration, and
rejects response paths outside the capture directory. Neither a precomputed safe
pose bank nor a reference floor plan enters exploration. The renderer's separate
reference file belongs exclusively to evaluation and reference visualization.

Body motion is kinematic on a level floor. Yaw/pitch targets and body turns have
explicit assumed rates; head settling and body travel count toward the simulated
action clock. Rendering, inference, map updates, and planning have separate wall
time measurements. This is not measured live robot-control latency.

## Mapping and movement

Before fusion, acquired depth points are filtered using the known articulated
robot proxy's oriented part bounds. This removes self observations of fingers,
arms, and body without discarding a blanket radius of nearby scene geometry.
The mask does not create free space behind the robot; other acquired views must
observe it. It uses robot calibration, not renderer instance labels.

An occupancy map projects the nearest observed obstacle at body height within
each azimuth bin. When no obstacle is observed, an acquired floor return limits
free-space clearing. Invalid depth never clears space. All observed hit endpoints
take precedence over free rays in an update. A conservative circular footprint
is inflated against both occupied and unknown grid cells, including cell area.
The map is a coarse 2.5D approximation: thin/missed obstacles, body articulation,
sloped floors, and dynamic objects need a richer collision model.

The first station acquires a full local sweep. Later stations order head views
toward unknown directions and also inspect overhead structure. Reachable
free/unknown boundary components provide goals; frontier size, path length, and
past station visits determine priority. Eight-connected grouping retains diagonal
visibility boundaries as frontier components. Paths remain four-connected to
prevent corner cutting. A soft clearance cost favors passage centers without
relaxing the hard footprint constraint; reported distance remains actual path
length. Only a bounded path prefix is executed before new observations.
This gives local inspection followed by exploration, not a proof that a room has
been fully reconstructed before departure. Returning through known passages is
allowed when other frontiers remain.

RGB-D surface points and visible-plus-typed learned edge points are voxel-fused
in world coordinates. Amodal-only predictions do not create observed 3D edges.
Distinct-image and translated-camera support counts remain separate. The output
is an observed colored point cloud and a structural edge cloud, with doorway gaps
left open. It is not a watertight building mesh or a ground-truth floor plan.

An optional postprocessor extracts tall Manhattan plane candidates and exports
15 cm quad patches only where acquired surface voxels have enough support.
Unsupported cells are never filled to complete a wall. Tall furniture can also
meet this geometric test, and a missing patch is not proof of a door. The OBJ is
an inspectable observed-patch model, not a semantic or watertight architecture.

Observed free-space clearance cores seed a provisional region graph. A watershed
splits wide areas across narrow observed connections. These are region and neck
candidates: furniture and missing observations can split a room, and wide portals
can merge rooms. The map does not receive reference room names or connectivity.

## Evaluation and limitations

The evaluator receives reference room bounds and architectural edges only after
the run. It measures observed edge coverage, predicted 3D edge precision and
completeness at 5/10 cm, room-entry order, portal crossings, and path clearance.
It separately checks every movement against the partial occupancy snapshot that
was available before movement. Reference furniture bounding-box checks are an
additional conservative audit, not a controller input or certified collision test.

Stopping on a budget or exhausted reachable frontiers is **partial completion**.
It cannot prove the entire building was observed. Areas hidden behind furniture
and surfaces outside acquired views remain absent. A first-building development
result is not a held-out generalization benchmark or a policy-superiority claim.
Future training/evaluation splits must group every view of a building together.

## Run and inspect

Use separate terminals for the warm renderer and the host perception process:

```bash
./installation/run_python.sh scripts/render_building.py \
  --config configs/buildings/connected_three_rooms.json \
  --output vis/multiroom/capture --requests vis/multiroom/requests --max-frames 1200

PYTHONPATH=src python scripts/run_multiroom.py \
  --output vis/multiroom/run_v4 --request-prefix v4_
PYTHONPATH=src python scripts/evaluate_multiroom.py \
  --results vis/multiroom/run_v4/experiment.json \
  --reference vis/multiroom/capture/reference.json
PYTHONPATH=src python scripts/check_multiroom_replay.py \
  --results vis/multiroom/run_v4/experiment.json --output vis/multiroom/run_v4/replay.json
PYTHONPATH=src python scripts/export_observed_mesh.py \
  --results vis/multiroom/run_v4/experiment.json
PYTHONPATH=src python scripts/make_multiroom_report.py \
  --results vis/multiroom/run_v4/experiment.json \
  --reference vis/multiroom/capture/reference.json --output vis/multiroom/report
PYTHONPATH=src python scripts/make_multiroom_topology.py \
  --run vis/multiroom/run_v4 --output vis/multiroom/report
PYTHONPATH=src python scripts/make_multiroom_report.py \
  --output vis/multiroom/report --refresh-page
ffmpeg -y -framerate 8 -i vis/multiroom/report/replay/frame_%04d.webp \
  -c:v libopenh264 -b:v 2600k -pix_fmt yuv420p -movflags +faststart -an \
  vis/multiroom/report/multiroom.mp4
touch vis/multiroom/requests/STOP
```

A fresh clone first needs the existing trained checkpoint and rendering assets;
see [perception setup](perception.md) and the repository installation instructions.
Use fresh output directories and unique request prefixes for repeated runs.
The server can resume an interrupted capture with `--resume`, verifying scene
provenance before processing pending requests. Full frames, predictions, raw
maps, PLY exports, and reports stay local under ignored `vis/`.

The existing network is reused without new training. The renderer remains warm,
only requested observations are generated, and small CPU maps drive exploration.
Training on connected-room doorway/corridor views can follow after this sensor
and exploration pipeline has been evaluated; its data must remain separated by
building rather than by image.

## Measured development result

The canonical local run is `vis/multiroom/run_v4/experiment.json`. It acquired
420 views at 42 stations, traveled 34.83 m, and first entered office → corridor →
living room → bedroom. All three physical doorways were crossed, with six total
crossings including revisits. It stopped at the configured image budget while
12 reachable frontiers remained; whole-building completion was not established.

| Reference region | Architectural edge visibility | Predicted edge completeness at 10 cm |
| --- | ---: | ---: |
| Office | 89.33% | 42.03% |
| Corridor | 97.01% | 71.38% |
| Living room | 91.23% | 47.77% |
| Bedroom | 79.88% | 45.98% |
| Entire building, length-weighted | 90.31% | 54.22% |

Fused predicted-edge precision at 10 cm was 64.82%. Completeness restricted to
reference edges actually seen was 59.92%; symmetric edge Chamfer was 19.79 cm.
Requiring the wall/floor/ceiling/door channel to match lowers precision to 41.37%
and completeness to 34.40%. Exploration coverage is ahead of learned structural
and semantic accuracy. These metrics are not comparable to the prior single-room
RGB-only cuboid dimension errors, and this building was used for development.

The shared map contains 108,861 observed surface voxels and 7,718 learned edge
voxels. Observed-only topology extraction found three substantial room candidates,
one corridor candidate, and three neck connections, plus 21 small free-space
fragments. All major regions still touch unknown cells. The optional wall export
contains 11 tall plane candidates and 8,700 supported patches; some candidates
may be furniture rather than architecture.

All 41 body movements stayed within their respective pre-move observed-safe maps.
The independent continuous 25 cm disk audit found no intersecting architectural
or furniture-proxy segments. Minimum remaining clearance was 20.27 cm against
architecture and 13.97 cm against recipe furniture bounding boxes. These checks
do not validate the entire humanoid body, downloaded-mesh collision shapes, or gait.

On DGX Spark / GB10, acquisition plus inference/mapping/planning took 419.71 s
wall time, separate from 1,379.08 s of simulated head/body movement and settling.
Median rendering was 702.06 ms/frame, model forward/sigmoid/probability transfer
7.49 ms/frame, self filtering plus map fusion 79.51 ms/frame, and frontier/path
planning 16.96 ms/station. The existing checkpoint was reused without training.
Rendering is the largest current acquisition cost; repeated full sweeps during
travel are another clear target for an adaptive observation budget.

The source passed 120 CPU unit tests and Ruff. An independent CPU replay verified
all acquired RGB/depth/probability/self-mask hashes and exactly reproduced 462 map
arrays across 42 stations, head targets, frontier choices, movement paths, action
clock, and final topology. Replay took 42.08 s without renderer, GPU inference, or
reference building access. `run_v4/replay.json` records the check. Earlier local
diagnostics exposed self-body occupancy, insufficient passage clearance preference,
and fragmented diagonal frontiers; regression tests cover the corrected behavior.

The next model experiment should train on diverse connected buildings and reserve
entire unseen buildings for validation/test. Prioritize doorway/corridor semantics
and uncertainty, then reduce redundant scans while preserving reconstruction
quality. A hierarchical per-room completion policy, realistic sensor/pose noise,
and full-body locomotion remain separate extensions.
