# RoomGraph

Synthetic indoor imagery and geometry-derived structural-edge annotations for
multi-view reconstruction. The first milestone is a calibrated Isaac Sim room
capture on NVIDIA DGX Spark. Training and reconstruction are subsequent milestones.

## Current scope

The smoke scene is a 6 × 5 × 3 metre empty room, with a doorway and an unglazed
window opening. Its 20 annotated edges describe the **interior architectural
shell** and inner opening contours. Exterior wall edges, opening reveal edges,
material boundaries, and mesh triangulation seams are excluded.

Four furnished variants now use this shell: bedroom/study, kitchen/dining, living
room, and shared office. Furniture is a separate layer, with procedural furnishings
and textured CC0 assets. Each scene has three calibrated perspective views and an
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
`top_down_overview.jpg` provide contact sheets. The four scenes produce 12
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
production dataset validation pipeline. These are four decorated versions of one
room shell, not yet connected multi-room buildings or industrial environments.

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
5. Extend the furnished-room prototype to connected rooms, garages, and factories.
6. Evaluate clutter, occlusion, held-out assets/layouts, and real-image transfer.

See [the design notes](docs/design.md) for scene organization and evaluation rules.
