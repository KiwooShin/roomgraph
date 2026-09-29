# RoomGraph

Synthetic indoor imagery and geometry-derived structural-edge annotations for
multi-view reconstruction. The first milestone is a calibrated Isaac Sim room
capture on NVIDIA DGX Spark. Training and reconstruction are subsequent milestones.

## Current scope

The smoke scene is a 6 × 5 × 3 metre empty room, with a doorway and an unglazed
window opening. Its 20 annotated edges describe the **interior architectural
shell** and inner opening contours. Exterior wall edges, opening reveal edges,
material boundaries, and mesh triangulation seams are excluded.

The architecture is independent of the renderer. Furnished scenes will add a
separate object layer, preserving paired empty/furnished geometry and camera poses.

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
./installation/run_python.sh scripts/render_smoke.py --output artifacts/smoke
```

This generates four overlapping calibrated views, each containing:

- RGB and an RGB/edge overlay.
- Visible structural edges and full projected structural edges as 8-bit PNG masks.
- Metric optical-axis depth and renderer-provided normals in a compressed NPZ file.
- Intrinsics and camera-to-world transforms in `manifest.json`.

After installing the development environment below, generate a portable HTML
report with an interactive structural graph, camera poses, and image comparisons:

```bash
.venv/bin/python scripts/make_report.py artifacts/smoke
```

Open `artifacts/smoke/report.html` directly in a browser; it needs no server or CDN.

The output also includes the USD scene and a contact sheet. An independent
ray/box calculation validates rendered depth at sparse pixel centers. The smoke
test fails if the 95th-percentile absolute error exceeds 3 cm in any view, or if
RGB/depth output is invalid. This is a geometric correctness gate, not a claim
about model accuracy or synthetic-to-real performance.

Ground-truth visibility currently uses analytic ray intersections with the demo's
opaque boxes. Imported furniture will require a general mesh visibility backend.
Dense metric edge sampling is suitable for the fixed demo camera distances; a
production exporter will use clipped, adaptive screen-space rasterization.

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
3. Implement dataset loading, edge/corner prediction, training, and evaluation.
4. Validate reconstruction with ground-truth edges, then predicted edges.
5. Add furnished kitchens, offices, bedrooms, garages, and industrial spaces.
6. Evaluate clutter, occlusion, held-out assets/layouts, and real-image transfer.

See [the design notes](docs/design.md) for scene organization and evaluation rules.
