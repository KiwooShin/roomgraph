# RoomGraph

Synthetic indoor imagery and geometry-derived structural-edge annotations for
multi-view reconstruction on NVIDIA DGX Spark. The project includes furnished
Isaac Sim captures, a trained structural-edge baseline, calibrated room fitting,
and sensor-assisted exploration of connected indoor spaces.

## Current scope

The smoke scene is a 6 × 5 × 3 metre empty room, with a doorway and an unglazed
window opening. Its 20 annotated edges describe the **interior architectural
shell** and inner opening contours. Exterior wall edges, opening reveal edges,
material boundaries, and mesh triangulation seams are excluded.

Four furnished variants now use this shell: bedroom/study, kitchen/dining, living
room, and shared office. Furniture is a separate layer, with procedural furnishings
and textured CC0 assets. Each scene has eight calibrated perspective views and an
orthographic top-down view with camera position/direction triangles.

## Installation on DGX Spark

Prerequisites: ARM64 Linux, Docker access, NVIDIA Container Toolkit, a compatible
NVIDIA driver, and ample free disk space. This setup builds Isaac Sim **5.1.0** at
commit `47d886f2858d1ceed556b21c88927aa67bc81c12`. GCC 11 and runtime dependencies
are installed inside a container; the host compiler is untouched.

Two compatibility fixes are included: the debug-draw library rename documented
in [upstream issue 303](https://github.com/isaac-sim/IsaacSim/issues/303), and a
container-only guard around ARM vector-math declarations that the bundled CUDA
compiler cannot parse. The source patch is idempotent; the Dockerfile checks the
expected header contents before applying its change.

```bash
./installation/build_isaac.sh
```

The upstream build prompts for NVIDIA's additional software terms on first use.
The source and persistent dependency cache are stored under
`~/.local/opt/indoor-structure`, configurable with `ROOMGRAPH_INSTALL_DIR`.
Build and runtime use the same container paths because Isaac's dependency links
can be absolute. Do not remove the cache while the installation is in use.

## Generate the smoke dataset

```bash
./installation/run_python.sh scripts/render_smoke.py --output vis/smoke
```

This generates four overlapping calibrated views, each containing:

- RGB and an RGB/edge overlay.
- Visible structural edges and full projected structural edges as 8-bit PNG masks.
- Metric optical-axis depth and renderer-provided normals in a compressed NPZ file.
- Intrinsics and camera-to-world transforms in `manifest.json`.

After installing the development environment below, generate a portable HTML
report with an interactive structural graph, camera poses, and image comparisons:

```bash
.venv/bin/python scripts/make_report.py vis/smoke
```

Open `vis/smoke/report.html` directly in a browser; it needs no server or CDN.
The `vis/` folder stays local and is excluded from Git. It contains the contact
sheet, per-view images, annotations, and the interactive report.

The output also includes the USD scene and a contact sheet. An independent
ray/box calculation validates rendered depth at sparse pixel centers. The smoke
test fails if the 95th-percentile absolute error exceeds 3 cm in any view, or if
RGB/depth output is invalid. This is a geometric correctness gate, not a claim
about model accuracy or synthetic-to-real performance.

Ground-truth visibility currently uses analytic ray intersections with the demo's
opaque boxes. The furnished exporter uses renderer depth instead.
Dense metric edge sampling is suitable for the fixed demo camera distances; a
production exporter will use clipped, adaptive screen-space rasterization.

## Furnished scenes and editable JSON

After installing the development environment below:

```bash
.venv/bin/python scripts/download_furnishing_assets.py
./installation/run_python.sh scripts/render_furnished.py --output vis/furnished
.venv/bin/python scripts/make_furnished_report.py vis/furnished
```

Open `vis/furnished/report.html` for the local gallery, RGB/structural-edge toggles,
furniture inventories, and top-down camera maps. `overview.jpg` and
`top_down_overview.jpg` provide contact sheets. The four scenes produce 32
perspective views at 1152 × 768, with 128 path-tracing samples per pixel by default.

Edit [configs/scenes](configs/scenes) to change furnishings, camera poses,
lighting, and render settings. See [the scene format](docs/scene-format.md).
Each output includes the input `scene.json`, a `manifest.json`, and a USD scene.
Assets download to ignored `assets/cache/`; their source URLs, CC0 license, and
checksums are recorded in [assets/catalog.json](assets/catalog.json).
Both `vis/` and downloaded assets remain local.

Furnished visibility uses rendered first-hit depth with a 2.5 cm tolerance.
Masks distinguish visible, hidden, and full projected architectural edges;
furniture edges and reflected structure in mirrors are excluded. This is a
prototype annotation method with pixel/silhouette approximation, not yet a
production dataset validation pipeline. This furnished dataset contains four
decorated versions of one room shell. The connected-building experiments below
use separate layouts and evaluation protocols.

## Development

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

Tests run without Isaac Sim. Renderer integration checks require the installation.
Generated data, caches, logs, and virtual environments are excluded from Git.

## Roadmap

1. Validate native GPU rendering, camera conventions, and structural annotations.
2. Parameterize empty layouts and capture trajectories; split data by building.
3. Extend the trained edge-detection pilot to diverse layouts and corner prediction.
4. Improve learned doorway semantics and connected-room edge reconstruction.
5. Extend the connected-space suite to garages, factories and larger floor plans.
6. Evaluate clutter, occlusion, held-out assets/layouts, and real-image transfer.

See [the design notes](docs/design.md) for scene organization and evaluation rules.

## Public gallery and moving-camera previews

The default furnished configs now contain eight camera views per room. A separate
visualization path moves the camera smoothly around an ellipse at 1.95 m height,
with 96 poses and geometry-derived overlays at every pose:

```bash
.venv/bin/python scripts/prepare_motion_configs.py
./installation/run_python.sh scripts/render_furnished.py \
  --configs artifacts/motion-configs/*.json --output vis/walkthrough
.venv/bin/python scripts/make_walkthrough_media.py vis/walkthrough /path/to/public/media
```

The motion preset renders at 768 × 512 with 32 samples per pixel. The compositor
adds a synchronized top-down marker and exports an 8-second MP4/GIF loop at 12 fps;
FFmpeg must be on `PATH`. This is a moving virtual camera, not a simulated robot
trajectory or a collision-validated navigation path.

To export compressed still previews and selected scene JSON for a website:

```bash
.venv/bin/python scripts/export_web_gallery.py vis/showcase /path/to/public/media
```

Exporting to a website is an explicit publication step. The full `vis/` captures,
USD scenes, depth arrays, and downloaded assets remain ignored in this repository.

## Trained head-camera perception pilot

The connected pipeline now generates room-disjoint training data, trains a
ResNet-18 edge model with TensorBoard tracking, and reconstructs a rectangular
room shell from predicted edges and known camera poses. Head-camera renders
include an original humanoid proxy for self-occlusion and mirror reflections.

Measured on 96 test images from four held-out synthetic room instances:
visible-edge F1 **0.916**, typed amodal-edge F1 **0.958**, and mean room-dimension
error **0.023 m**. These are controlled single-room results with known poses and
axes, not real-world robot reconstruction accuracy.

See [pipeline and commands](docs/perception.md),
[measured results](docs/results-headcam-v1.md),
[training efficiency and robustness](docs/training-efficiency.md), and
[training configuration](configs/training/headcam_v1.json).
The local gallery is `vis/perception/report.html`; training logs and checkpoints
are in `artifacts/runs/headcam_v1/`. All generated data and checkpoints stay local.

## Active head-camera pilot

The robot proxy can now turn its head independently of its body. A causal
three-second view planner tracks unseen directions and image-supported shell
hypotheses, then chooses the next head target. The local report combines furnished
RGB/edge overlays, top-down head direction, evidence maps, and comparisons with
a fixed scan and random trajectories.

See [the experiment protocol and limitations](docs/active-head-experiment.md).
The generated report is `vis/active_head/report/report.html`; all raw
visualizations stay local. This is a validation-room prototype with known poses
and translated bootstrap views, not autonomous navigation.

## Connected-room exploration

The multi-room pilot starts in a furnished office and expands an observed map
toward reachable unknown-space frontiers. It uses physically open passages,
articulated head views, a known-geometry robot self-mask, and short body movements.
Acquired RGB-D surfaces and learned visible structural edges accumulate in one
world frame; the controller has no reference floor plan or room destination list.

This explicitly adds ideal depth and known poses to the frozen RGB edge model.
It exports observed surface/edge point clouds and provisional region connections;
it does not fill unseen geometry or demonstrate dynamically walking hardware.
See [the protocol and reproducible commands](docs/multiroom-experiment.md).
The full local visualization is `vis/multiroom/report/report.html`.

## Multiple furnished spaces

Four distinct development buildings extend the connected-room pilot: a branching
home, loop-connected workspaces, an open office with side rooms, and a compact
apartment. Each uses the same frozen edge model, RGB-D sensor assumptions,
exploration policy and 300-image budget. Editable JSON specifies the architecture,
furnishings and start pose; the controller receives acquired observations only.

See [the suite protocol and commands](docs/space-suite-experiment.md). Run
`scripts/run_space_suite.py` to acquire, evaluate, replay-check and visualize the
cases sequentially. The aggregate local report is
`vis/space_suite/report/report.html`; full visualizations remain ignored by Git.
Room entry, observed edge visibility and learned reconstruction completeness are
reported separately, including cases that stop with unseen rooms or structure.
The [public comparison page](https://kiwooshin.github.io/roomgraph-spaces/) contains
selected moving-camera replays and interactive observed 3D maps.

A controlled follow-up tests bounded frontier-goal persistence with the same
layouts, checkpoint and 300-image budgets. It still enters 13 of 16 rooms, while
macro 3D completeness falls from 43.98% to 42.42% at 10 cm tolerance. The original
policy remains the default. The experimental adapter is
`scripts/run_space_suite_v2.py`; its paired report is
`vis/space_suite/policy_comparison/report.html`. Inspect both policies' paths,
overlays and observed maps in the
[published policy comparison](https://kiwooshin.github.io/roomgraph-spaces/#policy-comparison).

## Real RGB-D pilot: ARKitScenes

The first real-data experiment uses ARKitScenes validation video `42445021`,
visit `421380`: 240 chronologically sampled input views, the frozen synthetic
EdgeNet checkpoint, measured mobile depth, confidence maps and estimated camera
poses. It does not retrain the model or tune its 0.7 threshold. This is one
development capture, not a held-out real-world benchmark.

```bash
python scripts/download_arkitscenes.py
PYTHONPATH=src python scripts/run_arkitscenes.py
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' python scripts/check_arkitscenes_replay.py
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' python scripts/make_arkitscenes_report.py
```

Raw assets live under ignored `artifacts/real/arkitscenes/`. The self-contained
local report is `vis/arkitscenes/pilot_v2/report/report.html`, with a 48-second
recorded-camera replay, four rows of top-down + four overlays, interactive point
maps, and all error cases. The top-down video background is the final observed
map; its triangle and highlighted path show the recorded camera movement.

The adapter inverts ARKitScenes' world-to-camera axis-angle poses, interpolates
camera centers and rotations without extrapolation, converts millimeters to
meters, and normalizes image orientation using `sky_direction`. Calibration and
camera basis rotate together; tests verify unchanged 3D rays. RGB preprocessing
retains the existing model's anisotropic 384×256 resize with matching intrinsics.
No synthetic robot self-mask, assumed floor height, or rectangular-room fit is
applied. The earlier `pilot_v1` raw-orientation diagnostic and its source snapshot
remain local; it is superseded for model interpretation by the upright run.

The evaluator alone reads 139 laser-derived reference-depth images. Input
timestamps within 50 ms of those views are excluded. Reference points use nominal
ARKit calibration/poses for world placement; these poses are estimates, and the
reference is an incomplete set of surfaces. Thus the following are **surface
agreement scores against the available reference subset**, not independent
survey accuracy, architectural-edge accuracy, or whole-scene completeness.
Correct surfaces outside reference coverage can lower precision.

| Error condition | All valid depth: F1 at 10 cm | Confidence + multi-view: F1 at 10 cm |
| --- | ---: | ---: |
| Supplied calibration/poses | 67.24% | 86.42% |
| 2 cm / 1° independent pose jitter | 69.55% | 85.71% |
| 5 cm / 3° independent pose jitter | 56.10% | 74.74% |

Jitter standard deviations are per axis, averaged over three declared seeds.
The suite also tests a gradual 10 cm / 3° drift, +2% focal length, +2 pixel
principal point, +2% depth scale, and +50 ms pose timestamp offset. These are
illustrative sensitivity conditions, not measured 1X sensor specifications.
Some small biases improve this limited reference-agreement score; that is not
evidence that deliberately miscalibrating a camera improves actual geometry.

The conservative branch keeps high-confidence depth, requires surface support
from at least two images, and requires structural edges to have two images and
camera centers separated by at least 20 cm. At nominal poses it raises reference
precision from 50.72% to 76.96%, while reference recall falls from 99.68% to
98.53%. It retains only 226 of 3,427 learned edge voxels. There are no typed
architectural-edge labels here, so this does not establish improved edge accuracy.
Support counts and model confidence are not calibrated geometric uncertainty.

Inference runs once; all 24 fusion variants reuse its cached probabilities.
The complete experiment took 108.27 seconds, including preparation, inference,
reference construction and scoring, excluding download and media generation.
Median model-forward time was 7.66 ms. Exact CPU replay reproduced 24 arrays
across three selected maps, including a severe-jitter case.

Confidence filtering cannot repair systematic drift or calibration bias. Next
steps are robust overlapping-frame pose refinement with pose priors, loop-closure
validation, and re-fusion from original observations after corrections. Calibration
and time-offset refinement need sufficiently varied motion and constrained priors;
flat walls alone cannot constrain every parameter. Evaluate correction against
independent reference data before adoption. The RGB-only model does not consume
camera poses, so pose-noise augmentation alone cannot fix its 3D postprocessing.

Data and format sources: [ARKitScenes](https://github.com/apple-aiml-research/ARKitScenes),
[raw assets](https://github.com/apple-aiml-research/ARKitScenes/blob/main/raw/README.md),
[dataset license](https://github.com/apple-aiml-research/ARKitScenes/blob/main/LICENSE).
Follow dataset attribution and usage terms; downloaded assets remain local.

### Real-image preprocessing comparison

The frozen model now has a [same-view preprocessing diagnostic](docs/real-edge-preprocessing.md):
240 ARKitScenes views, 121 exact VGA matches, five input strategies, and a local
annotation review tool. Results remain qualitative until independently reviewed
structural-edge labels are available; no real-edge accuracy improvement is claimed.
